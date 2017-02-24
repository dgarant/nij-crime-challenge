import gzip
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler
import statsmodels
import statsmodels.discrete.discrete_model
import traceback
import warnings
import sys
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import os
import cPickle as pickle
import argparse
from scipy.special import gammaln
import scipy as sp
from sklearn import preprocessing as skpreprocessing
import sklearn

DAY_WINDOWS = [7, 14, 30, 61, 91]
OUTCOME_CATEGORIES = ["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "ALL"]

OUTCOME_VARS = []
for forecast_window_days in DAY_WINDOWS:
    for category in OUTCOME_CATEGORIES:
        OUTCOME_VARS.append("outcome_num_crimes_{0}days_{1}".format(forecast_window_days, category))

def main():
    cell_dim = 550

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", default="../../features/processed-features-{0}.csv.gz".format(cell_dim))
    args = parser.parse_args()

    meta_file = "../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim)
    meta = pd.read_csv(meta_file, index_col="id")
    total_crimes_by_cell = meta["num.crimes"].to_dict()

    param_groups, param_group_map = build_parameter_groups(meta, total_crimes_by_cell)

    features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim)))
    features = features.dropna()
    predictors = get_predictor_names(features)
    features["istrain"] = get_train_test_indicator(features)

    print("Training set size: {0}".format(features.istrain.sum()))
    print("Testing set size: {0}".format(features.shape[0] - features.istrain.sum()))
    transform = get_feature_preprocessor(features[features.istrain == 1], predictors, should_write=False)

    features["groupid"] = ["g{0}".format(param_group_map[cell_id]) 
                if cell_id in param_group_map else "i{0}".format(cell_id) 
                    for cell_id in features["cell_id"]]

    trans_preds = transform(features[predictors])
    pred_frame = pd.DataFrame(data=trans_preds, 
        columns=transform.final_feature_names)

    sub_features = features[["cell_id", "forecast_start", "groupid", "istrain"] + OUTCOME_VARS]
    sub_features.index = [x for x in range(len(sub_features))]
    pred_frame.index = [x for x in range(len(sub_features))]
    final_frame = pd.concat([sub_features, pred_frame], axis=1)
    final_frame = final_frame.sort_values(by=["groupid", "cell_id"])
    final_frame.to_csv(gzip.open(args.output_file, "w+"), index=False)
    
class FeaturePreprocessor(object):
    def __init__(self):
        #self.fourier_sampler = RBFSampler(random_state=10, n_components=50)
        self.scaler = skpreprocessing.StandardScaler()
        self.pca_transformer = PCA(n_components=70)

    def fit(self, features):
        self.original_feature_names = features.columns.values
        self.scaler.fit(features)
        scaled_data = self.scaler.transform(features)
        self.pca_transformer.fit(scaled_data)

        pca_output = self.pca_transformer.transform(scaled_data)
        print("Selected {0} principal components".format(pca_output.shape[1]))
        #self.fourier_sampler.fit(pca_output)
        #rbf_sampled = self.fourier_sampler.transform(pca_output)
        #final_features = statsmodels.tools.add_constant(rbf_sampled, prepend=True)
        self.final_feature_names = ["p_const"] + ["p_{0}".format(i) for i in range(pca_output.shape[1])]
    
    def __call__(self, features):
        scaled = self.scaler.transform(features)
        othog = self.pca_transformer.transform(scaled)
        #rbf_sampled = self.fourier_sampler.transform(othog)
        final_features = statsmodels.tools.add_constant(othog, prepend=True)
         
        return final_features


def generate_cell_features(path):
    """ 
        Assuming the file at `path` is ordered by cell_id, 
        iteratively produces dataframes representing each cell.
    """
    from io import StringIO

    with gzip.open(path, "rb") as handle:
        header = next(handle)
        prev_cell_id = None
        line_buffer = [header]
        for line in handle:
            components = line.split(",")
            cell_id = components[0]
            if cell_id != prev_cell_id and not prev_cell_id is None:
                str_f = StringIO(u"".join(line_buffer))
                yield pd.read_csv(str_f, index_col="cell_id")
                line_buffer = [header, line]
            line_buffer.append(line)
            prev_cell_id = cell_id



def load_potentials(meta, param_groups, param_group_map):
    group_cache = dict()
    for i, group in enumerate(param_groups):
        model_file = "../../models/mrf/nb-groups/{0}.p".format(i)
        with open(model_file, "r") as handle:
            group_cache[model_file] = pickle.load(handle)

    cell_models = dict()
    for cell_id in meta.index.values:
        cell_models[cell_id] = load_cell_potentials(cell_id, param_groups, param_group_map, group_cache)

    return cell_models

def get_pca_preds(ncomps):
    return ["p_const"] + ["p_{0}".format(x) for x in range(ncomps)]

def get_predictor_names(features):
    lagged_predictors = [p for p in features.columns.values if p.startswith("p_")]
    future_predictors = [p for p in features.columns.values if not p.startswith("outcome_") and 
                                                         not p.startswith("p_") and 
                                                         not p == "cell_id" and 
                                                         not p == "forecast_start" and 
                                                         not p == "groupid" and 
                                                         not p == "istrain"]
    predictors = lagged_predictors + future_predictors
    return predictors

def get_train_test_indicator(features):
    return features.apply(lambda x: int(hash(x["forecast_start"]) % 10 < 7), axis=1)

def get_train_test_split(features):
    """
        Creates a deterministic 70/30 train/test split from the supplied features.
    """
    indicator = get_train_test_indicator(features)
    train_features = features[indicator == 1]
    test_features = features[indicator == 0]
    return (train_features, test_features)


def get_feature_preprocessor(features, predictors, should_write=False):
    """ 
        Creates and returns a function which transforms features from their original 
        space to projected space in which the finite-dimensional dot product approximates RBF.

        Features are first scaled, then 'decorrelated' using PCA, then projected by 
        sampling from the fourier transform of the RBF kernel.
    """
    feature_transformer = FeaturePreprocessor()
    feature_transformer.fit(features[predictors])

    if should_write:
        with open("../../models/mrf/transformer.p", "wb+") as handle:
            pickle.dump(feature_transformer, handle)

    return feature_transformer

def load_feature_preprocessor():
    with open("../../models/mrf/transformer.p", "r") as handle:
        return pickle.load(handle)

def load_parameter_groups(path):
    """ Loads parameter trying groups from a file with columns 'group,cellid' """
    param_groups = [] # list of sets of parameter groups
    param_group_map = dict() # maps cell Id to the index of the group it participates in
    with open(path, "r") as handle:
        lines = handle.readlines()
        for line in lines[1:]:
            group, cellid = [int(a) for a in line.strip().split(",")]
            param_group_map[cellid] = group

            while len(param_groups) < (group+1):
                param_groups.append([])
            param_groups[group].append(cellid)
    return (param_groups, param_group_map)

def build_parameter_groups(meta, total_crimes_by_cell):

    target_file = "../../models/mrf/param_groups.csv"
    if os.path.exists(target_file):
        param_groups, param_group_map = load_parameter_groups(target_file)
    else:
        param_groups = [] # list of sets of parameter groups
        param_group_map = dict() # maps cell Id to the index of the group it participates in
        successors = dict()

        def can_group(cell_id):
            return (not cell_id in param_group_map)

        for i, row in meta.iterrows():
            successors[i] = [int(row[key]) for key in ["idnorth", "ideast", "idsouth", "idwest", 
                                         "idnortheast", "idnorthwest", 
                                         "idsoutheast", "idsouthwest"] if not pd.isnull(row[key])]

        def compatible(i, j):
            crime_match = False
            ic = total_crimes_by_cell[i]
            jc = total_crimes_by_cell[j]
            if ic > 1500 and jc > 1500:
                crime_match = True
            elif (ic > 500 and ic <= 1500 and jc > 500 and jc <= 1500):
                crime_match = True
            elif (ic > 100 and ic <= 500 and jc > 100 and jc <= 500):
                crime_match = True
            elif ic <= 100 and jc <= 100:
                crime_match = True

            return len(successors[i]) == len(successors[j]) and crime_match
            
        def bfs(root, succfun, node_limit=20):
            visited = set()
            visited.add(root)
            queue = [root]
            while queue:
                current = queue.pop(0)
                yield current
                for n in succfun(current):
                    if not n in visited and can_group(n) and compatible(root, n):
                        if len(visited) == node_limit:
                            return
                        visited.add(n)
                        queue.append(n)

        # most important groups are those with high # of crimes
        for i, row in meta.sort_values(by="num.crimes", ascending=False).iterrows():
            if can_group(i):

                # try to form a new group with BFS
                related_nodes = list(bfs(i, lambda k: successors[k]))
                if len(related_nodes) > 1:
                    for node in related_nodes: # includes i
                        param_group_map[node] = len(param_groups)
                    param_groups.append(related_nodes)

        # serialize the parameter typing groups

        with open("../../models/mrf/param_groups.csv", "w+") as handle:
            handle.write("group,cellid\n")
            for i,group in enumerate(param_groups):
                for member in group:
                    handle.write("{0},{1}\n".format(i, member))

    return (param_groups, param_group_map)

if __name__ == "__main__":
    main()
