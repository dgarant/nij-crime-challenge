from fit_potentials import *
import pandas as pd
import pyDPMP
import plmrf
import numpy as np
import json

cell_dim = 550

def main():

    print("loading features")
    features = pd.read_csv("../../models/poisson/forecast.csv".format(cell_dim), index_col="cell_id")
    train_features, test_features = get_train_test_split(features)
    test_dates = test_features["forecast_start"].unique()

    print("Loading parameter tying groups")
    param_group_file = "../../models/mrf/param_groups.csv"
    param_groups, param_group_map = load_parameter_groups(param_group_file)

    meta = pd.read_csv("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim), index_col="id")
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


    with open("../../models/mrf/forecast.csv", "w+") as handle:
        fieldnames = ["cell_id", "forecast_start"] + [o.replace(" ", ".") for o in OUTCOME_VARS] + \
            ["pred_" + o.replace(" ", ".") for o in OUTCOME_VARS]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for test_date in test_dates:
            cur_features = test_features[test_features["forecast_start"] == test_date]

            outcomes_by_cell = defaultdict(dict)

            for outcome_var in OUTCOME_VARS:
                if outcome_var in skip_outcomes:
                    continue

                cleaned_outcome_name = outcome_var.replace(" ", ".")

                conditioning_set = get_conditioning_set(cell_models, outcome_var, cur_features)
                dpmp_network = get_pydpmp_network(outcome_var, conditioning_set)
                nnodes = len(dpmp_network.nodes)
                nedges = len(dpmp_network.edges)
                if nnodes > 100 or not hasattr(dpmp_network, "weights"):
                    continue
                print("producing forecast for {0}, date {1}, {2} nodes, {3} edges".format(outcome_var, test_date, nnodes, nedges))
                if nnodes > 0 and nedges > 0:
                    forecast = produce_forecast(dpmp_network)
                    print(forecast)
                elif nnodes > 0 and nedges == 0:
                    forecast = dict()
                    for name, val in conditioning_set.items():
                        if "_pred_" in name:
                            cell_id = int(name.split("_")[0])
                            forecast[cell_id] = val
                else:
                    print("using empty forecast!")
                    forecast = dict()

                cell_mrf_forecast = dict()
                for cell_outcome, exp_val in forecast.items():
                    cell_id = cell_outcome.split("_", 1)[0]
                    cell_mrf_forecast[cell_id] = exp_val
        
                unique_cell_ids = list(set(features.index.values))
                orig_preds = test_features.loc[unique_cell_ids]["pred_" + cleaned_outcome_name]
                actual_outcomes = test_features.loc[unique_cell_ids][cleaned_outcome_name]
                for i, cell_id in enumerate(unique_cell_ids):
                    outcomes_by_cell[cell_id]["cell_id"] = cell_id
                    outcomes_by_cell[cell_id]["forecast_start"] = test_date
                    if cell_id in cell_mrf_forecast:
                        outcomes_by_cell[cell_id]["pred_" + cleaned_outcome_name] = cell_mrf_forecast[cell_id]
                    else:
                        outcomes_by_cell[cell_id]["pred_" + cleaned_outcome_name] = orig_preds[i]
                    outcomes_by_cell[cell_id][cleaned_outcome_name] = actual_outcomes[i]
            
            for cell_id, record in outcomes_by_cell.items():
                writer.writerow(record)

def get_conditioning_set(cell_models, outcome_var, features):
    cell_conditions = dict()
    for cell_id, row in features.iterrows():
        model = cell_models[cell_id][outcome_var]

        cleaned_outcome = outcome_var.replace(" ", ".")
        if model["model_type"] == "poisson":
            cname = "{0}_pred_{1}".format(cell_id, cleaned_outcome)
            expval = row["pred_" + cleaned_outcome]
            cell_conditions[cname] = expval
        elif model["model_type"] == "median":
            cname = "{0}_{1}".format(cell_id, cleaned_outcome)
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

def produce_forecast(dpmp_network):
    proposal = pyDPMP.proposals.random_walk_proposal_1d(1)
    initial_particles = dict([(v, np.repeat(0, 20)) for v in dpmp_network.nodes])

    def cb(x):
        print("finished iteration")

    xMAP, xParticles, stats = pyDPMP.DPMP_infer(dpmp_network, initial_particles,
                                     20, # num paricles
                                     proposal,
                                     pyDPMP.particleselection.SelectDiverse(),
                                     pyDPMP.messagepassing.MaxSumMP(dpmp_network),
                                     conv_tol=None,
                                     max_iters=10,
                                     callback=cb)
    return xMAP

if __name__ == "__main__":
    main()
