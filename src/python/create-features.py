from collections import defaultdict, OrderedDict
import csv
import gzip
import calendar
import datetime
from dateutil import parser as dparser
from dateutil.relativedelta import relativedelta

cell_dimension = 550

cell_ids = set()
with open("../../models/cells/cells-dim-{0}-ids.csv".format(cell_dimension), "r") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        cell_ids.add(row["id"])
    
# (cellid, date) -> (category) -> count
cell_count_index = defaultdict(lambda: defaultdict(int))

start_date = None
end_date = None
category_abbrevs = {
    "BURGLARY-PROPERTY CRIME" : "burg_prop",
    "BURGLARY-SUSPICIOUS" : "burg_susp",
    "MOTOR VEHICLE THEFT-PROPERTY CRIME" : "mvt_property",
    "OTHER-DISORDER" : "other_disorder",
    "OTHER-NON CRIMINAL/ADMIN" : "other_noncrim",
    "OTHER-PERSON CRIME" : "other_person",
    "OTHER-PROPERTY CRIME" : "other_property",
    "OTHER-SUSPICIOUS" : "other_susp",
    "OTHER-TRAFFIC" : "other_traffic",
    "STREET CRIMES-DISORDER" : "street_disorder",
    "STREET CRIMES-PERSON CRIME" : "street_person"
}
crime_categories = set(category_abbrevs.values())
outcome_categories = set(["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "OTHER"])
outcome_index = defaultdict(lambda: dict(zip(outcome_categories, [0] * 4)))

with gzip.open("../../features/raw_crimes_cells_{0}.csv.gz".format(cell_dimension)) as gzfile:
    reader = csv.DictReader(gzfile)
    
    for i, row in enumerate(reader):
        cell_count_index[(row["id"], row["occ_date"])][category_abbrevs[row
        ["CATEGORY2"]]] += 1
        outcome_index[(row["id"], row["occ_date"])][row["CATEGORY"]] += 1
        
        if start_date is None or row["occ_date"] < start_date:
            start_date = row["occ_date"]
        if end_date is None or row["occ_date"] > end_date:
            end_date = row["occ_date"]
       
# Generates all dates between from_date and to_date, inclusive
def date_generator(from_date, to_date):
    while from_date <= to_date:
        yield from_date
        from_date = from_date + datetime.timedelta(days=1)      
   

def crimes_in_window(cellid, start_date, end_date, cindex):
    crime_dict = defaultdict(int)
    for d in date_generator(start_date, end_date):
        str_d = d.isoformat()
        for category, val in cindex[(cellid, str_d)].items():
            crime_dict[category] += val
    return crime_dict  
        
features = []

with gzip.open("../../features/full-features.csv.gz", "w+") as feature_file:
    for forecast_start in date_generator(dparser.parse(start_date).date(), dparser.parse(end_date).date()):
        # try to eliminate some autocorrelation by 
        # keeping only the 1st and 15th as starting dates
        if forecast_start.day != 1 and forecast_start.day != 15:
            continue
            
        writer = None
            
        for cellid in cell_ids:
            cur_features = OrderedDict()
            cur_features["cell_id"] = cellid
            cur_features["forecast_start"] = forecast_start.isoformat()
            
            def store_outcomes(outcome_dict, time_window):
                for category in outcome_categories:
                    cur_features["outcome_num_crimes_{0}_{1}".format(time_window, category)] = outcome_dict[category]
            
            # outcomes
            for day_window in [7, 14, 30, 61]:
                edate = forecast_start + datetime.timedelta(days=day_window)
                cur_outcomes = crimes_in_window(cellid, forecast_start, edate, outcome_index)
                store_outcomes(cur_outcomes, "{0}days".format(day_window))
            
            def store_count_features(count_dict, time_window):
                for category in crime_categories:
                    cur_features["p_num_crimes_{0}_{1}".format(time_window, category)] = count_dict[category]
            
            # daily counts
            for delta_days in range(3, 8):
                ddate = forecast_start + datetime.timedelta(days=-delta_days)
                counts_by_category = crimes_in_window(cellid, ddate, ddate, cell_count_index)
                store_count_features(counts_by_category, "{0}days_ago".format(delta_days))
            
            # weekly counts
            for delta_weeks in range(4):
                window_start = forecast_start + datetime.timedelta(days=-(delta_weeks * 7 + 3))
                window_end = forecast_start + datetime.timedelta(days=-(delta_weeks * 7 + 9))
                counts_by_category = crimes_in_window(cellid, window_start, window_end, cell_count_index)
                store_count_features(counts_by_category, "{0}weeks_ago".format(delta_weeks+1))
                
            # monthly counts
            for delta_months in range(4):
                window_start = forecast_start + datetime.timedelta(days=-(delta_months * 30 + 3))
                window_end = forecast_start + datetime.timedelta(days=-(delta_months * 30 + 32))
                counts_by_category = crimes_in_window(cellid, window_start, window_end, cell_count_index)
                store_count_features(counts_by_category, "{0}months_ago".format(delta_months+1))
                
            # last year's values  
            py_start_date = forecast_start - relativedelta(years=1)
            for day_window in [7, 14, 30, 61]:
                py_end = py_start_date + datetime.timedelta(days=6)
                py_outcomes = crimes_in_window(cellid, py_start_date, py_end, cell_count_index)
                store_count_features(py_outcomes, "py{0}days".format(day_window))
            
            # day structure
            for day_window in [7, 14, 30, 61]:
                window_end = forecast_start + datetime.timedelta(days=day_window)
                num_thursdays = 0
                num_fridays = 0
                num_saturdays = 0
                num_tuesdays = 0
                for d in date_generator(forecast_start, window_end):
                    if d.weekday() == 5:
                        num_saturdays += 1
                    if d.weekday() == 4:
                        num_fridays += 1
                    if d.weekday() == 3:
                        num_thursdays += 1
                    if d.weekday() == 2:
                        num_tuesdays += 1
                cur_features["num_saturdays_next{0}days".format(day_window)] = num_saturdays
                cur_features["num_fridays_next{0}days".format(day_window)] = num_fridays
                cur_features["num_thursdays_next{0}days".format(day_window)] = num_thursdays
                cur_features["num_tuesdays_next{0}days".format(day_window)] = num_tuesdays
                
            # TODO: weather, events
            
            if not writer:
                writer = csv.DictWriter(feature_file, fieldnames=cur_features.keys())
                writer.writeheader()
            writer.writerow(cur_features)
            #for key, val in cur_features.items():
            #    print("{0}: {1}".format(key, val))