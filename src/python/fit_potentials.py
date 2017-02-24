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
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from process_features import *

def safe_mkdir(dir_to_create):
    try:
        os.makedirs(dir_to_create)
    except OSError:
        pass

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
        features = pd.read_csv(gzip.open("../../features/processed-features-{0}.csv.gz".format(cell_dim), "rb"), 
            index_col="cell_id", nrows=args.nrows)
    else:
        features = pd.read_csv(gzip.open("../../features/processed-features-{0}.csv.gz".format(cell_dim), "rb"), 
            index_col="cell_id")
    features = features.dropna()
    predictors = get_predictor_names(features)

    train_features = features[features.istrain == 1]
    test_features = features[features.istrain == 0]

    del features

    # cells without any crime aren't considered later in the process
    sample_features = train_features
    global_models = dict()
    global_train_preds = np.asarray(sample_features[predictors])
    print(global_train_preds)
    for outcome_var in OUTCOME_VARS:
        outcome_dev = np.std(sample_features[outcome_var])
        print("Fitting global model of {0} response std dev {1}".format(outcome_var, np.round(outcome_dev, 2)))
        if outcome_dev < 1:
            cur_train_preds = global_train_preds[:, 0:8]
        else:
            cur_train_preds = global_train_preds

        print("Using {0} predictors".format(cur_train_preds.shape[1]))
        model = statsmodels.discrete.discrete_model.Poisson(
            sample_features[outcome_var], global_train_preds)
        global_models[outcome_var] = model.fit(maxiter=1000, disp=1)

    all_errors = []
    for i, group in enumerate(param_groups):
        target_file = "../../models/mrf/nb-groups/{0}.p".format(i)
        if not os.path.exists(target_file) or args.no_cache:
            if  i % args.num_jobs != args.job_id:
                #print("Skipping group {0}, another job is doing this".format(i))
                continue

            print("J{0}: Fitting model for group {1}, with members {2}{3}".format(args.job_id, i, group, 
                ", NRMSE: {0}".format(np.mean(all_errors)) if len(all_errors) > 0 else "" ))
            errors, models = fit_count_models(train_features, test_features, group, global_models)
            with open(target_file, "wb+") as handle:
                pickle.dump(models, handle)
            all_errors.extend(errors)

    for i, cell_id in enumerate(set(meta.index.values)):
        if cell_id in param_group_map:
            continue

        target_file = "../../models/mrf/nb-individuals/{0}.p".format(cell_id)
        if not os.path.exists(target_file) or args.no_cache:
            if  i % args.num_jobs != args.job_id:
                #print("Skipping cell {0}, another job is doing this".format(cell_id))
                continue
            print("J{0}: Fitting model for {1}, with {2} total crimes{3}".format(args.job_id, cell_id, total_crimes_by_cell[cell_id],
                ", NRMSE: {0}".format(np.mean(all_errors)) if len(all_errors) > 0 else "" ))
            errors, models = fit_count_models(train_features, test_features, cell_id, global_models)
            with open(target_file, "wb+") as handle:
                pickle.dump(models, handle)

            all_errors.extend(errors)
            if len(all_errors) > 0 and i % 1000 == 0:
                print("-------------------------------------")
                print("Mean NRMSE: {0}".format(np.mean(all_errors)))
                print("-------------------------------------")

    print("Mean NRMSE: {0}".format(np.mean(all_errors)))

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

def load_cell_potentials(cell_id, param_groups, param_group_map, cache=None):
    if cell_id in param_group_map:
        model_file = "../../models/mrf/nb-groups/{0}.p".format(param_group_map[cell_id])
    else:
        model_file = "../../models/mrf/nb-individuals/{0}.p".format(cell_id)

    if not cache is None and model_file in cache:
        return cache[model_file]

    with open(model_file, "r") as handle:
        return pickle.load(handle)

def fit_count_models(train_features, test_features, cell_ids, global_models):
    """ Fits regression models to the crime counting process """

    outcome_params = dict()
    cell_data = train_features.loc[cell_ids]
    n_comps = 70 if cell_data.shape[0] > 100 else 25

    predictors = get_pca_preds(n_comps)
    cur_predictors = np.asarray(cell_data[predictors])

    cell_test_data = test_features.loc[cell_ids]
    cell_test_preds = np.asarray(cell_test_data[predictors])

    errors = []
    for outcome_var in OUTCOME_VARS:
        cur_responses = cell_data[outcome_var]
        test_responses = cell_test_data[outcome_var]
        domain = [np.min(cur_responses), np.max(cur_responses)]
        bw_estimates = calc_bandwidth_estimates(cur_responses)

        if cur_responses.sum() < 10:
            outcome_params[outcome_var] = {"model_type" : "median", 
                    "median_value" : np.median(cur_responses), "domain" : domain, "bw" : bw_estimates}
        else:
            print("\trunning {0}, {1}, response sum is {2}, training set shape: {3}".format(cell_ids, outcome_var, cur_responses.sum(), cell_test_preds.shape))
            try:
                pmodel = statsmodels.discrete.discrete_model.NegativeBinomial(cur_responses, cell_predictors) 
                start_params = global_models[outcome_var].params
                results = pmodel.fit_regularized(method="l1", 
                        #start_params=np.append(poisson_fit.params, 0.1), 
                        start_params = start_params,
                        maxiter=300, disp=0, acc=1e-4, skip_hessian=True)

                pred_resp = pmodel.predict(cell_test_preds)
                assert not np.isnan(results.params.max())

                #lmodel = sklearn.linear_model.LinearRegression(fit_intercept=False)
                #results = lmodel.fit(transformed_predictors, cur_responses)
                #pred_resp = lmodel.predict(transformed_test_predictors)
                #outcome_params[outcome_var] = {"model_type" : "linear-reg", 
                #    "parameters" : results.coef_, 
                #    "mle_result" : None, "domain" : domain, "bw" : bw_estimates}

                if np.mean(cur_responses) > 10:
                    rmse = np.sqrt(np.sum(np.power(test_responses - pred_resp, 2)))
                    nrmse = rmse / np.mean(cur_responses)
                    errors.append(nrmse)

                outcome_params[outcome_var] = {"model_type" : "negative-binomial", 
                    "parameters" : results.params,
                    "mle_result" : results.mle_retvals, "domain" : domain, 
                    "bw" : bw_estimates, "n_features" : len(predictors)}
            except (ConvergenceWarning, np.linalg.linalg.LinAlgError, AssertionError) as e:
                if cur_responses.sum() >= 100:
                    sys.stderr.write("convergence problem with lots of non-zero cases (cells: {0})!\n".format(cell_ids))
                outcome_params[outcome_var] = {"model_type" : "median", 
                    "median_value" : np.median(cur_responses), "domain" : domain, "bw" : bw_estimates}

    return (errors, outcome_params)

def calc_bandwidth_estimates(vals):
    """ Calculates rule-of-thumb bandwidth for Gaussian kernel based on squared distances """
    s1 = np.random.permutation(vals)
    s2 = np.random.permutation(vals)
    probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    points = sp.stats.mstats.mquantiles(np.power(s1 - s2, 2.0), prob=probs)
    return dict(zip(probs, points))

if __name__ == "__main__":
    main()

