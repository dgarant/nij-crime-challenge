import plmrf
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
from ipdb import launch_ipdb_on_exception
import os
import cPickle as pickle

def safe_mkdir(dir_to_create):
    try:
        os.makedirs(dir_to_create)
    except OSError:
        pass

DAY_WINDOWS = [7, 14, 30, 61, 91]
OUTCOME_CATEGORIES = ["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "ALL"]

def main():
    cell_dim = 550

    meta_file = "../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim)
    meta = pd.read_csv(meta_file, index_col=0)
    total_crimes_by_cell = meta["num.crimes"].to_dict()

    outcome_vars = []
    for forecast_window_days in DAY_WINDOWS:
        for category in OUTCOME_CATEGORIES:
            outcome_vars.append("outcome_num_crimes_{0}days_{1}".format(forecast_window_days, category))

    features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim), "rb"), index_col="cell_id")
    features = features.dropna()
    lagged_predictors = [p for p in features.columns.values if p.startswith("p_")]
    future_predictors = [p for p in features.columns.values if not p.startswith("outcome_") and 
                                                         not p.startswith("p_") and 
                                                         not p == "cell_id" and 
                                                         not p == "forecast_start"]

    predictors = lagged_predictors + future_predictors

    # this sampler gives projection into a feature space which approximates RBF
    fourier_sampler = RBFSampler(random_state=10)
    fourier_sampler.fit(features[predictors])
    pca_transformer = PCA(n_components=0.99)
    pca_transformer.fit(features[predictors])
    print("num principal components: {0}".format(len(pca_transformer.explained_variance_)))

    def transform(dat):
        pca_data = pca_transformer.transform(dat)
        fourier_features = fourier_sampler.transform(dat)
        features_with_constant = statsmodels.tools.add_constant(fourier_features)
        return features_with_constant

    param_group, param_groups = build_parameter_groups(meta, total_crimes_by_cell)

    sample_features = features.sample(n=5000)
    global_models = dict()
    for outcome_var in outcome_vars:
        print("Fitting global model of {0}".format(outcome_var))
        model = statsmodels.discrete.discrete_model.NegativeBinomial(
            sample_features[outcome_var], transform(sample_features[predictors]))
        global_models[outcome_var] = model.fit(maxiter=1000, disp=0)

    for i, group in enumerate(param_groups):
        target_file = "../../models/mrf/nb-groups/{0}.p".format(i)
        if not os.path.exists(target_file):
            print("Fitting model for group {0}, with members {1}".format(i, group))
            models = fit_count_models(features, group, transform, outcome_vars, predictors)
            with open(target_file, "wb+") as handle:
                pickle.dump(models, handle)

    for cell_id in set(features.index.values):
        if cell_id in param_groups:
            continue
        target_file = "../../models/mrf/nb-individuals/{0}.p".format(cell_id)

        if not os.path.exists(target_file):
            print("Fitting model for {0}, with {1} total crimes".format(cell_id, total_crimes_by_cell[cell_id]))
            models = fit_count_models(features, cell_id, transform, outcome_vars, predictors)
            with open(target_file, "wb+") as handle:
                pickle.dump(models, handle)

def fit_count_models(features, cell_ids, transform, outcome_vars, predictors):
    """ Fits regression models to the crime counting process """

    outcome_params = dict()
    cell_data = features.loc[cell_ids]
    cur_predictors = cell_data[predictors]
    transformed_predictors = transform(cur_predictors)

    for outcome_var in outcome_vars:
        cur_responses = cell_data[outcome_var]
        if cur_responses.sum() < 10:
            outcome_params[outcome_var] = {"model_type" : "zero"}
        else:
            print("running {0}, {1}, response sum is {2}".format(cell_ids, outcome_var, cur_responses.sum()))
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    pmodel = statsmodels.discrete.discrete_model.NegativeBinomial(cur_responses, transformed_predictors) 
                    start_params = global_models[outcome_var].params
                    results = pmodel.fit_regularized(method="l1", 
                            #start_params=np.append(poisson_fit.params, 0.1), 
                            start_params = start_params,
                            maxiter=200, disp=0, skip_hessian=True, acc=1e-4)

                    #print(results.mle_retvals)
                    assert not np.isnan(results.params.max())

                    outcome_params[outcome_var] = {"model_type" : "negative-binomial", 
                        "parameters" : results.params, 
                        "mle_result" : results.mle_retvals}
                except (ConvergenceWarning, np.linalg.linalg.LinAlgError, AssertionError) as e:
                    if cur_responses.sum() >= 100:
                        raise ValueError("convergence problem with lots of non-zero cases!")
                    else:
                        outcome_params[outcome_var] = {"model_type" : "median", 
                            "median_value" : np.median(cur_responses)}

    return outcome_params

def build_parameter_groups(meta, total_crimes_by_cell):

    # non-inclusive limits on the number of crimes occuring 
    # within a cell to participate in parameter tying
    group_crime_bounds = [9, 500] 
    param_groups = [] # list of sets of parameter groups
    param_group = dict() # maps cell Id to the index of the group it participates in

    successors = dict()

    def can_group(cell_id):
        info = meta.loc[cell_id]
        return (not cell_id in param_group and 
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
                    and len(successors[n]) == len(nearby_nodes) - 1 and not n in param_group]
            if len(related_nodes) > 1:
                for node in related_nodes: # includes i
                    param_group[node] = len(param_groups)
                param_groups.append(related_nodes)

    # serialize the parameter typing groups
    safe_mkdir("../../models/mrf/")
    safe_mkdir("../../models/mrf/nb-groups/")
    safe_mkdir("../../models/mrf/nb-individuals/")

    with open("../../models/mrf/param_groups.csv", "w+") as handle:
        handle.write("group,cellid\n")
        for i,group in enumerate(param_groups):
            for member in group:
                handle.write("{0},{1}\n".format(i, member))

    return (param_groups, param_group)

if __name__ == "__main__":
    main()

