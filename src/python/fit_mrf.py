import plmrf
import gzip
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
    print("Loading parameter tying groups")
    param_group_file = "../../models/mrf/param_groups.csv"
    param_groups, param_group_map = load_parameter_groups(param_group_file)

    meta = pd.read_csv("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim), index_col="id")
    print("Loading cell potentials")
    cell_models = load_potentials(meta, param_groups, param_group_map)
    #print("Loading training data")
    #features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim), "rb"), index_col="cell_id", nrows=10)
    #train_data, test_data = get_train_test_split(features)
    feature_transformer = load_feature_preprocessor()
    transformed_predictor_names = feature_transformer.final_feature_names

    outcome_networks = dict()
    for outcome in OUTCOME_VARS:
        print("Working on {0}".format(outcome))
        outcome_networks[outcome] = build_network(meta, outcome, transformed_predictor_names, cell_models, param_groups, param_group_map)
        with open("../../models/mrf/mrf-structure_{0}.p".format(outcome), "w+") as handle:
            pickle.dump(outcome_networks[outcome], handle)

def build_network(meta, outcome_var, predictor_names, cell_models, param_groups, param_group_map):

    feature_transformer = load_feature_preprocessor()

    potential_funs = []
    variable_spec = []
    tied_weights = [[]] * len(param_groups)
    conditioned = dict()
    for cell_id, cell_meta in meta.iterrows():
        model = cell_models[cell_id][outcome_var]
        cell_meta = meta.loc[cell_id]
        if cell_id % 1000 == 0:
            print("\tWorking on cell {0}".format(cell_id))

        cell_outcome_name = "{0}_{1}".format(cell_id, outcome_var)

        domain = np.arange(model["domain"][0], model["domain"][1]+1)
        variable_spec.append(plmrf.VariableDef(cell_outcome_name, ddomain=domain))

        cell_predictor_names = ["{0}_{1}".format(cell_id, p) for p in predictor_names]
        
        # form connections to all cells with greater ids, 
        # to ensure they are created only once
        adj_cell_ids = [int(cell_meta[k]) for k in ["idnorth", "idsouth", "ideast", "idwest", 
                    "idnortheast", "idsoutheast", "idnorthwest", "idsouthwest"] 
                    if not cell_meta[k] is None and cell_id < cell_meta[k]]

        for adj_cell_id in adj_cell_ids:
            adj_potential = plmrf.GaussianPotential([cell_outcome_name, "{0}_{1}".format(adj_cell_id, outcome_var)], bandwidth=5)
            potential_funs.append(adj_potential)

            if cell_id in param_group_map:
                tied_weights[param_group_map[cell_id]].append(len(potential_funs) - 1)

        if model["model_type"] == "negative-binomial":
            new_potential = NBPotential(cell_outcome_name, cell_predictor_names, 
                model["parameters"], feature_transformer, bandwidth=model["bw"][0.5])
            conditioned[cell_outcome_name] = cell_predictor_names
        elif model["model_type"] == "median":
            new_potential = plmrf.GaussianPotential([cell_outcome_name], 
                location=model["median_value"], bandwidth=model["bw"][0.5])
            conditioned[cell_outcome_name] = []
        else:
            raise ValueError("Unknown model_type: {0}".format(model["model_type"]))

        potential_funs.append(new_potential)
        if cell_id in param_group_map:
            tied_weights[param_group_map[cell_id]].append(len(potential_funs) - 1)

    network = plmrf.LogLinearMarkovNetwork(potential_funs, variable_spec, tied_weights=tied_weights, conditioned=conditioned)
    
    return network

class NBPotential(plmrf.PotentialFunction):

    def __init__(self, response_var, predictors, model_params, feature_transformer, bandwidth):
        self.response_var = response_var
        self.predictors = predictors
        self.model_params = model_params
        self.feature_transformer = feature_transformer
        self.bandwidth = bandwidth
    
    def variables(self):
        return [self.response_var]

    def __call__(self, dmap):
        features = np.column_stack([dmap[p] for p in predictors])
        transformed_features = self.feature_transformer(features)
        expected_val = np.exp(transformed_features, np.dot(self.model_params[:-1]))

        response = dmap[self.response_var]

        return np.exp(-np.power(expected_val - response, 2.0) / self.bandwidth)

    def __str__(self):
        return "NBPotential(" + self.response_var + ")"


if __name__ == "__main__":
    main()

