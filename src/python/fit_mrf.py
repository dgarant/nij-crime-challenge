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
    param_group_file = "../../models/mrf/param_groups.csv"
    param_groups, param_group_map = load_parameter_groups(param_group_file)

    meta = pd.read_csv("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim), index_col="id")
    cell_models = load_potentials(meta, param_groups, param_group_map)
    
class NBPotential(plmrf.PotentialFunction):

    def __init__(self, response_var, predictors, model_params, feature_transformer)
        self.response_var = response_var
        self.predictors = predictors
        self.model_params = model_params
        self.feature_transformer = feature_transformer
    
    def variables(self):
        return [self.response_var] + self.predictors

    def __call__(self, dmap):
        features = np.column_stack([dmap[p] for p in predictors])
        transformed_features = self.feature_transformer(features)
        response = dmap[self.response_var]
        return nb_mass(response, transformed_features, self.model_params)

if __name__ == "__main__":
    main()

