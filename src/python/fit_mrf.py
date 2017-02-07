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
    #cell_models = load_potentials(meta, param_groups, param_group_map)
    cell_models = None
    print("Loading training data")
    features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim), "rb"), index_col="cell_id")
    train_data, test_data = get_train_test_split(features)
    predictor_names = get_predictor_names(train_data)

    outcome_networks = dict()
    for outcome in OUTCOME_VARS:
        print("Working on {0}".format(outcome))
        outcome_networks[outcome] = build_network(meta, train_data, outcome, predictor_names, cell_models, param_groups, param_group_map)

    with open("../../models/mrf/mrf-structure.p", "w+") as handle:
        pickle.dump(outcome_networks, handle)

def build_network(meta, train_data, outcome_var, predictor_names, cell_models, param_groups, param_group_map):

    feature_transformer = load_feature_preprocessor()

    potential_funs = []
    variable_spec = []
    tied_weights = [[]] * len(param_groups)
    for cell_id, cell_meta in meta.iterrows():
        #model = cell_models[cell_id][outcome_var]
        model = load_cell_potentials(cell_id, param_groups, param_group_map)[outcome_var]
        cell_meta = meta.loc[cell_id]
        if cell_id % 100 == 0:
            print("\tWorking on cell {0}".format(cell_id))

        cell_outcome_name = "{0}_{1}".format(cell_id, outcome_var)

        cell_data = train_data.loc[cell_id]
        cell_responses = cell_data[outcome_var]
        domain = [np.min(cell_responses), np.max(cell_responses)]

        variable_spec.append(plmrf.VariableDef(cell_outcome_name, ddomain=domain))

        cell_predictor_names = ["{0}_{1}".format(cell_id, p) for p in transformed_predictor_names]
        
        # form connections to all cells with greater ids, 
        # to ensure they are created only once
        adj_cell_ids = [int(cell_meta[k]) for k in ["idnorth", "idsouth", "ideast", "idwest", 
                    "idnortheast", "idsoutheast", "idnorthwest", "idsouthwest"] if not cell_meta[k] is None and cell_id < cell_meta[k]]
        for adj_cell_id in adj_cell_ids:
            adj_potential = plmrf.GaussianPotential([cell_outcome_name, "{0}_{1}".format(adj_cell_id, outcome_var)], bandwidth=5)
            potential_funs.append(adj_potential)

            if cell_id in param_group_map:
                tied_weights[param_group_map[cell_id]].append(len(potential_funs) - 1)

        if model["model_type"] == "negative-binomial":
            new_potential = NBPotential(cell_outcome_name, cell_predictor_names, model["parameters"], feature_transformer)
        elif model["model_type"] == "median":
            new_potential = plmrf.GaussianPotential([cell_outcome_name], location=model["median_value"], bandwidth=20)
        else:
            raise ValueError("Unknown model_type: {0}".format(model["model_type"]))

        potential_funs.append(new_potential)
        if cell_id in param_group_map:
            tied_weights[param_group_map[cell_id]].append(len(potential_funs) - 1)

    network = plmrf.LogLinearMarkovNetwork(potential_funs, variable_spec, tied_weights)
    
class NBPotential(plmrf.PotentialFunction):

    def __init__(self, response_var, predictors, model_params, feature_transformer):
        self.response_var = response_var
        self.predictors = predictors
        self.model_params = model_params
        self.feature_transformer = feature_transformer

        # seems reasonable, maybe should tune this
        self.bandwidth = 5 
    
    def variables(self):
        return [self.response_var]

    def __call__(self, dmap):
        features = np.column_stack([dmap[p] for p in predictors])
        transformed_features = self.feature_transformer(features)
        expected_val = np.exp(transformed_features, np.dot(self.model_params[:-1]))

        response = dmap[self.response_var]

        # laplace kernel
        return np.exp(-np.abs(expected_val - response) / self.bandwidth)

    def __str__(self):
        return "NBPotential(" + self.response_var + ")"


if __name__ == "__main__":
    main()

