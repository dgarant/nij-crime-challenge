
from process_features import *
from fit_potentials import *
import scipy as sp

def main():
    cell_dim = 550
    param_group_file = "../../models/mrf/param_groups.csv"
    param_groups, param_group_map = load_parameter_groups(param_group_file)

    meta = pd.read_csv("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim), index_col="id")

    transform = load_feature_preprocessor()

    nrmse_vals = []
    for cell_data in generate_cell_features("../../features/count-features-{0}.csv.gz".format(cell_dim)):
        cell_id = cell_data.index.values[0]
        train_data, test_data = get_train_test_split(cell_data)
        potentials = load_cell_potentials(cell_data.index.values[0], param_groups, param_group_map)
        predictors = get_predictor_names(cell_data)
        for nrmse in evaluate_models(test_data, potentials, cell_id, transform, predictors):
            nrmse_vals.append(nrmse)
            if len(nrmse_vals) % 10 == 0:
                print("----Mean NRMSE: {0}".format(np.mean(nrmse_vals)))
 
def evaluate_models(features, outcome_params, test_cell_ids, transform, predictors):
    test_data = features.loc[test_cell_ids]
    cur_predictors = test_data[predictors]
    transformed_predictors = transform(cur_predictors)

    for outcome_var in OUTCOME_VARS:
        cur_responses = test_data[outcome_var]
        model_spec = outcome_params[outcome_var]
        domain = np.arange(cur_responses.max())
        pred_responses = None
        if model_spec["model_type"] == "poisson":
            params = model_spec["parameters"]

            if np.mean(cur_responses) > 10:
                pmodel = statsmodels.discrete.discrete_model.Poisson(cur_responses, transformed_predictors) 
                pred_responses = pmodel.predict(params)
        elif model_spec["model_type"] == "linear-reg":
            if np.mean(cur_responses) > 10:
                params = model_spec["parameters"]
                pred_responses = np.dot(transformed_predictors, params)

        if not pred_responses is None:
            diffs = pred_responses - cur_responses
            rmse = np.sqrt(np.mean(np.power(diffs, 2)))
            nrmse = rmse / np.mean(cur_responses)

            print("Cell {0}\t\t{1}\t\tNRMSE {2}\tMean Resp: {3}\tEMean: {4}\tBias: {5}".format(
                test_cell_ids, outcome_var, np.round(nrmse, 2), np.round(np.mean(cur_responses), 2), 
                np.round(np.mean(pred_responses), 2), np.round(np.mean(cur_responses - pred_responses), 2)))
            
            yield nrmse

if __name__ == "__main__":
    main()

