import plmrf
import gzip
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.decomposition import PCA

cell_dim = 550

meta_file = "../../models/cells/cells-dim-{0}-meta.csv".format(cell_dim)
meta = pd.read_csv(meta_file, index_col=0)
total_crimes_by_cell = meta["num.crimes"].to_dict()

DAY_WINDOWS = [7, 14, 30, 61, 91]
OUTCOME_CATEGORIES = ["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "ALL"]

outcome_vars = []
for forecast_window_days in DAY_WINDOWS:
    for category in OUTCOME_CATEGORIES:
        outcome_vars.append("outcome_num_crimes_{0}days_{1}".format(forecast_window_days, category))

features = pd.read_csv(gzip.open("../../features/count-features-{0}.csv.gz".format(cell_dim), "rb"), 
                       index_col="cell_id", nrows=20000)
features = features.dropna()
print(features)
lagged_predictors = [p for p in features.columns.values if p.startswith("p_")]
future_predictors = [p for p in features.columns.values if not p.startswith("outcome_") and 
                                                     not p.startswith("p_") and 
                                                     not p == "cell_id" and 
                                                     not p == "forecast_start"]

for cell_id in features.index.values:
    cell_data = features.loc[cell_id]
    for outcome_var in outcome_vars:
        cur_responses = cell_data[outcome_var]
        print(cur_responses)
        cur_predictors = cell_data[lagged_predictors+future_predictors]
        print(cur_predictors)
        # remove linear dependencies amongst features only

def bfs(root, succfun, node_limit=30):
    visited = set()
    queue = [root]
    while queue:
        current = queue.pop(0)
        yield current
        for n in succfun(root):
            if not n in visited:
                if len(visited) == node_limit:
                    return
                visited.add(n)
                queue.append(n)

# non-inclusive limits on the number of crimes occuring 
# within a cell to participate in parameter tying
group_crime_bounds = [0, 500] 
param_groups = [] # list of sets of parameter groups
param_group = dict() # maps cell Id to the index of the group it participates in

successors = dict()
for i, row in meta.iterrows():
    successors[i] = [int(row[key]) for key in ["idnorth", "ideast", "idsouth", "idwest", 
                                 "idnortheast", "idnorthwest", 
                                 "idsoutheast", "idsouthwest"] if not pd.isnull(row[key])]

for i, row in meta.iterrows():
    if not i in param_group and row["num.crimes"] > group_crime_bounds[0] and row["num.crimes"] < group_crime_bounds[1]:
        # try to form a new group
        nearby_nodes = list(bfs(i, lambda k: successors[k]))
        related_nodes = [n for n in nearby_nodes 
            if np.abs(total_crimes_by_cell[n] - total_crimes_by_cell[i] < 10)
                and len(successors[n]) == len(nearby_nodes) - 1 and not n in param_group]
        for node in related_nodes: # includes i
            param_group[node] = len(param_groups)
        param_groups.append(set(related_nodes))

#for i, g in enumerate(param_groups):
#    print("{0}: {1}".format(i, g))
#    for m in g:
#        print("\t{0} -> {1}".format(m, param_group[m]))
#print(param_groups)


#for row in reader:
#    print(row)

