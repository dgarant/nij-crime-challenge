import plmrf
import gzip
import json
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os
import cPickle as pickle
import statsmodels
import statsmodels.discrete.discrete_model
import argparse
from fit_potentials import *
from scipy.special import gammaln

cell_dim = 550

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", default=False, action="store_true")
    parser.add_argument("--num-jobs", default=1, type=int)
    parser.add_argument("--job-id", default=0, type=int)
    args = parser.parse_args()

    print("Loading parameter tying groups")
    param_group_file = "../../models/mrf/param_groups.csv"
    param_groups, param_group_map = load_parameter_groups(param_group_file)

    meta = pd.read_csv("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim), index_col="id")

    print("Loading training data")
    features = pd.read_csv("../../models/poisson/forecast.csv".format(cell_dim), index_col="cell_id")
    train_features, tset_features = get_train_test_split(features)

    print("Loading cell potentials")
    with open("../../models/poisson/models.json", "r") as handle:
        models = json.load(handle)

    cell_models = dict()
    for cell_id in set(features.index.values):
        if cell_id in param_group_map:
            cell_models[cell_id] = models["g{0}".format(param_group_map[cell_id])]
        else:
            cell_models[cell_id] = models["i{0}".format(cell_id)]

    skip_outcomes = set(["outcome_num_crimes_7days_BURGLARY"])

    for i, outcome in enumerate(OUTCOME_VARS):
        if outcome in skip_outcomes or i % args.num_jobs != args.job_id:
            continue

        print("J{0} Working on {1}".format(args.job_id, outcome))
        structure_file = "../../models/mrf/mrf-structure_{0}.p".format(outcome)
        if os.path.exists(structure_file) and not args.no_cache:
            print("Loading network ...")
            with open(structure_file, "rb") as handle:
                network = pickle.load(handle)
        else:
            print("Building network ...")
            network = build_network(meta, outcome, cell_models, param_groups, param_group_map)
            with open(structure_file, "wb+") as handle:
                pickle.dump(network, handle, pickle.HIGHEST_PROTOCOL)

        print("Building training data map ...")
        train_map = build_training_data(train_features, outcome, cell_models)
        print("Fitting network ...")
        fit_result = network.fit(train_map, log=True, maxiter=100)
        print(fit_result)
        with open(structure_file, "wb+") as handle:
            pickle.dump(network, handle, pickle.HIGHEST_PROTOCOL)

def build_training_data(train_features, outcome_var, cell_models):

    train_data_map = dict()

    grouped_cells = train_features.groupby(level=0)

    for cell, cell_features in grouped_cells:
        cell_features = cell_features.sort_values(by="forecast_start")
        cleaned_outcome_var = outcome_var.replace(" ", ".")
        outcome_name = "{0}_{1}".format(cell, cleaned_outcome_var)
        train_data_map[outcome_name] = np.array(cell_features[cleaned_outcome_var])

        model = cell_models[cell][outcome_var]
        if model["model_type"] == "poisson":
            pred_name = "{0}_pred_{1}".format(cell, cleaned_outcome_var)
            train_data_map[pred_name] = cell_features["pred_{0}".format(cleaned_outcome_var)]

    return train_data_map

def build_network(meta, outcome_var, cell_models, param_groups, param_group_map):

    feature_transformer = load_feature_preprocessor()

    potential_funs = []
    variable_spec = []
    tied_weights = defaultdict(list)
    conditioned = dict()
    for cell_id, cell_meta in meta.iterrows():
        model = cell_models[cell_id][outcome_var]
        cell_meta = meta.loc[cell_id]
        if cell_id % 1000 == 0:
            print("\tWorking on cell {0}".format(cell_id))

        cleaned_outcome_var = outcome_var.replace(" ", ".")
        cell_outcome_name = "{0}_{1}".format(cell_id, cleaned_outcome_var)
        predictor_name = "{0}_pred_{1}".format(cell_id, cleaned_outcome_var)

 
        adj_cell_ids = [int(cell_meta[k]) for k in ["idnorth", "idsouth", "ideast", "idwest", 
                    "idnortheast", "idsoutheast", "idnorthwest", "idsouthwest"] 
                    if not pd.isnull(cell_meta[k])]
        any_var_adj = len([i for i in adj_cell_ids if cell_models[i][outcome_var]["model_type"] != "median"]) != 0
        if not any_var_adj:
            pass # if this node isn't connected to any others, don't include it in the MRF
        elif model["model_type"] in ["negative-binomial", "poisson"]:
            domain = np.arange(min(model["domain"]), max(model["domain"])+1)

            variable_spec.append(plmrf.VariableDef(cell_outcome_name, ddomain=domain))

            # form connections to all cells with greater ids, 
            # to ensure they are created only once
            for adj_cell_id in [a for a in adj_cell_ids if cell_id < a]:
                adj_potential = plmrf.GaussianPotential([cell_outcome_name, 
                    "{0}_{1}".format(adj_cell_id, cleaned_outcome_var)], bandwidth=5)
                potential_funs.append(adj_potential)

                if cell_id in param_group_map:
                    tied_weights[param_group_map[cell_id]].append(len(potential_funs) - 1)

            adj_median_cell_outcomes = ["{0}_{1}".format(i, cleaned_outcome_var) for i in adj_cell_ids 
                if cell_models[i][outcome_var]["model_type"] == "median"]

            new_potential = plmrf.GaussianPotential(
                [cell_outcome_name, predictor_name], bandwidth=max(model["bw"]["50%"][0], 1.0))
            conditioned[cell_outcome_name] = [predictor_name] + adj_median_cell_outcomes
            potential_funs.append(new_potential)

            if cell_id in param_group_map:
                tied_weights[param_group_map[cell_id]].append(len(potential_funs) - 1)

        elif model["model_type"] == "median":
            pass
        else:
            raise ValueError("Unknown model_type: {0}".format(model["model_type"]))

    unwrapped_tied_weights = tied_weights.values()
    network = plmrf.LogLinearMarkovNetwork(potential_funs, variable_spec, 
        tied_weights=unwrapped_tied_weights, conditioned=conditioned)
    
    return network

if __name__ == "__main__":
    main()

