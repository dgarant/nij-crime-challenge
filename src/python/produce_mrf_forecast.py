from fit_potentials import *
import pandas as pd
import pyDPMP
import plmrf
import numpy as np

cell_dim = 550

def main():

    print("loading features")
    features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim), "rb"), index_col="cell_id")
    train_features, test_features = get_train_test_split(features)
    test_dates = test_features["forecast_start"].unique()
    feature_transformer = load_feature_preprocessor()

    print("Loading parameter tying groups")
    param_group_file = "../../models/mrf/param_groups.csv"
    param_groups, param_group_map = load_parameter_groups(param_group_file)

    meta = pd.read_csv("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim), index_col="id")
    print("Loading cell potentials")
    cell_models = load_potentials(meta, param_groups, param_group_map)

    for test_date in test_dates:
        cur_features = test_features[test_features["forecast_start"] == test_date]
        for outcome_var in OUTCOME_VARS:
            conditioning_set = get_conditioning_set(cell_models, outcome_var, cur_features, feature_transformer)
            dpmp_network = get_pydpmp_network(outcome_var, conditioning_set)
            nnodes = len(dpmp_network.nodes)
            nedges = len(dpmp_network.edges)
            print("producing forecast for {0}, date {1}, {2} nodes, {3} edges".format(outcome_var, test_date, nnodes, nedges))
            if nnodes > 0 and nedges > 0:
                forecast = produce_forecast(dpmp_network)
                print(forecast)
            elif nnodes > 0 and nedges == 0:
                forecast = dict()
                for name, val in conditioning_set.items():
                    if name.endswith("_nbexp"):
                        cell_id = int(name.split("_")[0])
                        forecast[cell_id] = val
            else:
                print("using empty forecast!")
                forecast = dict()

            save_forecast(cell_models, outcome_var, forecast, test_date)

def save_forecast(cell_models, outcome_var, forecast, test_date):
    cell_value = dict()
    for cell_id, outcome_models in cell_models.items():
        model = outcome_models[outcome_var]
        if model["model_type"] in ["negative-binomial", "poisson"]:
            cname = "{0}_{1}".format(cell_id, outcome_var)
            cell_value[cell_id] = forecast[cname]
        elif model["model_type"] == "median":
            cell_value[cell_id] = model["median_value"]
    
    basename = "{0}_{1}".format(outcome_var, test_date)
    with open("../../models/mrf/forecasts/{0}.csv".format(basename), "w+") as handle:
        handle.write("cell_id,forecasted_crimes\n")
        for cell_id, value in cell_value.items():
            handle.write("{0},{1}\n".format(cell_id, value))

def get_conditioning_set(cell_models, outcome_var, features, feature_transformer):
    cell_conditions = dict()
    for cell_id, row in features.iterrows():
        cell_preds = np.array([row[feature_transformer.original_feature_names]])
        model = cell_models[cell_id][outcome_var]
        if model["model_type"] == "negative-binomial":
            cname = "{0}_{1}_nbexp".format(cell_id, outcome_var)
            trans_features = feature_transformer(cell_preds)
            expval = np.exp(np.dot(trans_features, model["parameters"][:-1]))[0]
            cell_conditions[cname] = expval
        elif model["model_type"] == "median":
            cname = "{0}_{1}".format(cell_id, outcome_var)
            cell_conditions[cname] = model["median_value"]

    return cell_conditions

def get_pydpmp_network(outcome_var, conditioning_vals):
    """ 
        Creates a PyDPMP network which is consistent structure of the 
        plmrf network for the given outcome variable 
    """

    structure_file = "../../models/mrf/mrf-structure_{0}.p".format(outcome_var)
    with open(structure_file, "rb") as handle:
        network = pickle.load(handle)

    var_names = set([v.name for v in network.variable_spec])

    node_potentials = dict()
    edge_potentials = dict()
    weight_dict = dict()

    full_weight_list = network._LogLinearMarkovNetwork__expand_weights(network.weights)

    for i, p in enumerate(network.potential_funs):
        potential_variables = tuple(sorted(p.variables()))
        outcome_vars = list(set(p.variables()).intersection(var_names))

        if len(outcome_vars) == len(potential_variables):
            edge_potentials[potential_variables] = p
            weight_dict[potential_variables] = full_weight_list[i]
        else:
            node_potentials[outcome_vars[0]] = p
            weight_dict[outcome_vars[0]] = full_weight_list[i]

    def node_pot(s, vs):
        data = {s : vs}
        for v in network.conditioned[s]:
            data[v] = conditioning_vals[v]
        val = node_potentials[s](data) * weight_dict[s]
        return val

    def edge_pot(s, t, vs, vt):
        return edge_potentials[(s, t)]({s: vs, t: vt}) * weight_dict[(s, t)]

    dpmp_network = pyDPMP.mrf.MRF(node_potentials.keys(),
                                  edge_potentials.keys(),
                                  node_pot, edge_pot)
    return dpmp_network

def produce_indep_forecast(cell_models, outcome_var):
    for cell_id
    cell_models

def produce_forecast(dpmp_network):
    proposal = pyDPMP.proposals.random_walk_proposal_1d(1)
    initial_particles = dict([(v, np.repeat(0, 20)) for v in dpmp_network.nodes])
    xMAP, xParticles, stats = pyDPMP.DPMP_infer(dpmp_network, initial_particles,
                                     20, # num paricles
                                     proposal,
                                     pyDPMP.particleselection.SelectDiverse(),
                                     pyDPMP.messagepassing.MaxSumMP(dpmp_network),
                                     conv_tol=None,
                                     max_iters=10)
    return xMAP

if __name__ == "__main__":
    main()
