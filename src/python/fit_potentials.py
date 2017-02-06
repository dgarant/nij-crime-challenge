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

def safe_mkdir(dir_to_create):
    try:
        os.makedirs(dir_to_create)
    except OSError:
        pass

DAY_WINDOWS = [7, 14, 30, 61, 91]
OUTCOME_CATEGORIES = ["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "ALL"]

OUTCOME_VARS = []
for forecast_window_days in DAY_WINDOWS:
    for category in OUTCOME_CATEGORIES:
        OUTCOME_VARS.append("outcome_num_crimes_{0}days_{1}".format(forecast_window_days, category))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", default=False, action="store_true")
    parser.add_argument("--num-jobs", default=1, type=int)
    parser.add_argument("--job-id", default=0, type=int)
    parser.add_argument("--nrows", default=None, type=int)
    args = parser.parse_args()

    safe_mkdir("../../models/mrf/")
    safe_mkdir("../../models/mrf/nb-groups/")
    safe_mkdir("../../models/mrf/nb-individuals/")

    cell_dim = 550

    meta_file = "../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim)
    meta = pd.read_csv(meta_file, index_col="id")
    total_crimes_by_cell = meta["num.crimes"].to_dict()

    param_groups, param_group_map = build_parameter_groups(meta, total_crimes_by_cell)

    if not args.nrows is None:
        features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim), "rb"), 
            index_col="cell_id", nrows=args.nrows)
    else:
        features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim), "rb"), 
            index_col="cell_id")
    predictors = get_predictor_names(features)

    train_features, test_features = get_train_test_split(features)
    print("Training set size: {0}".format(train_features.shape))
    print("Testing set size: {0}".format(test_features.shape))
    transform = get_feature_preprocessor(train_features, predictors, should_write=args.job_id == 0)

    sample_features = train_features.sample(n=min(train_features.shape[0], 20000))
    global_models = dict()
    for outcome_var in OUTCOME_VARS:
        print("Fitting global model of {0}".format(outcome_var))
        model = statsmodels.discrete.discrete_model.NegativeBinomial(
            sample_features[outcome_var], transform(sample_features[predictors]))
        global_models[outcome_var] = model.fit(maxiter=1000, disp=0)

    for i, group in enumerate(param_groups):
        target_file = "../../models/mrf/nb-groups/{0}.p".format(i)
        if not os.path.exists(target_file) or args.no_cache:
            if  i % args.num_jobs != args.job_id:
                #print("Skipping group {0}, another job is doing this".format(i))
                continue

            print("J{0}: Fitting model for group {1}, with members {2}".format(args.job_id, i, group))
            models = fit_count_models(train_features, group, transform, predictors, global_models)
            with open(target_file, "wb+") as handle:
                pickle.dump(models, handle)

    for i, cell_id in enumerate(set(features.index.values)):
        if cell_id in param_group_map:
            continue

        target_file = "../../models/mrf/nb-individuals/{0}.p".format(cell_id)
        if not os.path.exists(target_file) or args.no_cache:
            if  i % args.num_jobs != args.job_id:
                #print("Skipping cell {0}, another job is doing this".format(cell_id))
                continue
            print("J{0}: Fitting model for {1}, with {2} total crimes".format(args.job_id, cell_id, total_crimes_by_cell[cell_id]))
            models = fit_count_models(train_features, cell_id, transform, predictors, global_models)
            with open(target_file, "wb+") as handle:
                pickle.dump(models, handle)

def get_predictor_names(features):
    lagged_predictors = [p for p in features.columns.values if p.startswith("p_")]
    future_predictors = [p for p in features.columns.values if not p.startswith("outcome_") and 
                                                         not p.startswith("p_") and 
                                                         not p == "cell_id" and 
                                                         not p == "forecast_start"]
    predictors = lagged_predictors + future_predictors
    return predictors

def get_train_test_split(features):
    """
        Creates a deterministic 70/30 train/test split from the supplied features.
    """
    features = features.dropna()
    # deterministic train/test sampling 
    row_hash = features.apply(lambda x: hash(tuple(x)) % 10, axis=1)
    train_features = features[row_hash < 7]
    test_features = features[row_hash >= 7]
    return (train_features, test_features)

def get_feature_preprocessor(features, predictors, should_write=False):
    """ 
        Creates and returns a function which transforms features from their original 
        space to projected space in which the finite-dimensional dot product approximates RBF.

        Features are first 'decorrelated' using PCA, then projected by 
        sampling from the fourier transform of the RBF kernel.
    """
    fourier_sampler = RBFSampler(random_state=10, n_components=100)
    pca_transformer = PCA(n_components=0.99)
    pca_transformer.fit(features[predictors])

    pca_output = pca_transformer.transform(features[predictors])
    fourier_sampler.fit(pca_output)

    if should_write:
        with open("../../models/mrf/transformer.p", "wb+") as handle:
            pickle.dump({"pca" : pca_transformer, "rbf-sampler" : fourier_sampler}, handle)

    def transform(dat):
        return pca_rbf_transform(pca_transformer, fourier_sampler, dat)

    return transform

def nb_mass(response, features, params):
    """ Probability mass of response observations under negative binomial model """
    pred_resp = np.exp(np.dot(features, params[:-1]))
    alpha = np.exp(params[-1])
    
    size = 1 / alpha
    prob = size / (size + pred_resp)
    mass = np.exp(gammaln(size+response) - gammaln(response+1) + size*np.log(prob) + response*np.log(1-prob))
    return mass

def nb_map(features, domain, params):
    maps = []
    alpha = np.exp(params[-1])
    size = 1 / alpha
    pred_resp = np.exp(np.dot(features, params[:-1]))
    for l in pred_resp:
        prob = size / (size + l)
        mass = np.exp(gammaln(size+domain) - gammaln(domain+1) + size*np.log(prob) + domain*np.log(1-prob))
        maps.append(domain[np.argmax(mass)])
    return maps

def pca_rbf_transform(pca_transformer, fourier_sampler, dat):
    pca_data = pca_transformer.transform(dat)
    fourier_features = fourier_sampler.transform(pca_data)
    features_with_constant = statsmodels.tools.add_constant(fourier_features)
    return features_with_constant

def load_feature_preprocessor():
    with open("../../models/mrf/transformer.p", "r") as handle:
        transformers = pickle.load(handle)
        pca_transformer = transformers["pca"]
        fourier_sampler = transformers["rbf-sampler"]

    def transform(dat):
        return pca_rbf_transform(pca_transformer, fourier_sampler, dat)

    return transform

def load_cell_potentials(cell_id, param_groups, param_group_map, cache=None):
    if cell_id in param_group_map:
        model_file = "../../models/mrf/nb-groups/{0}.p".format(param_group_map[cell_id])
    else:
        model_file = "../../models/mrf/nb-individuals/{0}.p".format(cell_id)

    if not cache is None and model_file in cache:
        return cache[model_file]

    with open(model_file, "r") as handle:
        return pickle.load(handle)

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

def fit_count_models(features, cell_ids, transform, predictors, global_models):
    """ Fits regression models to the crime counting process """

    outcome_params = dict()
    cell_data = features.loc[cell_ids]
    cur_predictors = cell_data[predictors]
    transformed_predictors = transform(cur_predictors)

    for outcome_var in OUTCOME_VARS:
        cur_responses = cell_data[outcome_var]
        if cur_responses.sum() < 10:
            outcome_params[outcome_var] = {"model_type" : "median", "median_value" : np.median(cur_responses)}
        else:
            print("\trunning {0}, {1}, response sum is {2}".format(cell_ids, outcome_var, cur_responses.sum()))
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    pmodel = statsmodels.discrete.discrete_model.NegativeBinomial(cur_responses, transformed_predictors) 
                    start_params = global_models[outcome_var].params
                    print(start_params)
                    results = pmodel.fit_regularized(method="l1", 
                            #start_params=np.append(poisson_fit.params, 0.1), 
                            start_params = start_params,
                            maxiter=300, disp=0, acc=1e-4)

                    #print(results.mle_retvals)
                    assert not np.isnan(results.params.max())

                    outcome_params[outcome_var] = {"model_type" : "negative-binomial", 
                        "parameters" : results.params, 
                        "mle_result" : results.mle_retvals}
                except (ConvergenceWarning, np.linalg.linalg.LinAlgError, AssertionError) as e:
                    if cur_responses.sum() >= 100:
                        sys.stderr.write("convergence problem with lots of non-zero cases (cells: {0})!\n".format(cell_ids))
                    outcome_params[outcome_var] = {"model_type" : "median", 
                        "median_value" : np.median(cur_responses)}

    return outcome_params

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

    # non-inclusive limits on the number of crimes occuring 
    # within a cell to participate in parameter tying
    group_crime_bounds = [9, 500] 

    target_file = "../../models/mrf/param_groups.csv"
    if os.path.exists(target_file):
        param_groups, param_group_map = load_parameter_groups(target_file)
    else:
        param_groups = [] # list of sets of parameter groups
        param_group_map = dict() # maps cell Id to the index of the group it participates in
        successors = dict()

        def can_group(cell_id):
            info = meta.loc[cell_id]
            return (not cell_id in param_group_map and 
                info["num.crimes"] > group_crime_bounds[0] and 
                info["num.crimes"] < group_crime_bounds[1])

        for i, row in meta.iterrows():
            successors[i] = [int(row[key]) for key in ["idnorth", "ideast", "idsouth", "idwest", 
                                         "idnortheast", "idnorthwest", 
                                         "idsoutheast", "idsouthwest"] if not pd.isnull(row[key])]

        def bfs(root, succfun, node_limit=30):
            visited = set()
            visited.add(root)
            queue = [root]
            while queue:
                current = queue.pop(0)
                yield current
                for n in succfun(root):
                    if not n in visited and can_group(n):
                        if len(visited) == node_limit:
                            return
                        visited.add(n)
                        queue.append(n)

        for i, row in meta.iterrows():
            if can_group(i):

                # try to form a new group with BFS
                nearby_nodes = list(bfs(i, lambda k: successors[k]))

                related_nodes = [n for n in nearby_nodes 
                    if np.abs(total_crimes_by_cell[n] - total_crimes_by_cell[i] < 10)
                        and len(successors[n]) == len(nearby_nodes) - 1 and not n in param_group_map]
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

