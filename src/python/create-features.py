from collections import defaultdict, OrderedDict
import csv
import gzip
import calendar
import datetime
from dateutil import parser as dparser
from dateutil.relativedelta import relativedelta
import sys, os
import pandas as pd
import gzip

cell_dimension = 550
DAY_WINDOWS = [7, 14, 30, 61, 91]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", default=1, type=int)
    parser.add_argument("--job-id", default=0, type=int)
    args = parser.parse_args()

    feature_file = "../../features/count-features-{0}.csv.gz{1}".format(
        cell_dimension, str(args.job_id) if args.num_jobs > 1 else "")

    process(feature_file, args.job_id, args.num_jobs)

def process(feature_file, job_id, njobs):
    cell_ids = set()
    with open("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dimension), "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cell_ids.add(row["id"])
        
    # (cellid, date) -> (category) -> count
    cell_count_index = defaultdict(lambda: defaultdict(int))

    data_start_date = None
    data_end_date = None
    crime_categories = set(["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "OTHER"])
    outcome_categories = set(["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "ALL"])
    outcome_index = defaultdict(lambda: defaultdict(int))

    with gzip.open("../../features/raw_crimes_cells_{0}.csv.gz".format(cell_dimension)) as gzfile:
        reader = csv.DictReader(gzfile)
        
        for i, row in enumerate(reader):
            cell_count_index[(row["id"], row["occ_date"])][row["CATEGORY"]] += 1
            outcome_index[(row["id"], row["occ_date"])][row["CATEGORY"]] += 1
            
            if data_start_date is None or row["occ_date"] < data_start_date:
                data_start_date = row["occ_date"]
            if data_end_date is None or row["occ_date"] > data_end_date:
                data_end_date = row["occ_date"]   
           
# Generates all dates between from_date and to_date, inclusive
    def date_generator(from_date, to_date):
        while from_date <= to_date:
            yield from_date
            from_date = from_date + datetime.timedelta(days=1)      
       
    def crimes_in_window(cellid, start_date, end_date, cindex):
        if start_date < data_start_date or end_date > data_end_date:
            crime_dict = defaultdict(lambda: None)
            return crime_dict

        crime_dict = defaultdict(int)
        for d in date_generator(start_date, end_date):
            str_d = d.isoformat()
            if (cellid, str_d) in cindex:
                for category in cindex[(cellid, str_d)]:
                    crime_dict[category] += cindex[(cellid, str_d)][category]
                    crime_dict["ALL"] += cindex[(cellid, str_d)][category]
        return crime_dict  
          
    data_start_date = dparser.parse(data_start_date).date()
    data_end_date = dparser.parse(data_end_date).date()

    with gzip.open(feature_file, "w+") as handle:
        for cellid in sorted(cell_ids, key=lambda c: int(c)):

            if int(cellid) % 100 == 0:
                print("{0}: Working on cell {1}".format(datetime.datetime.now().isoformat(), cellid))

            if int(cellid) % njobs != job_id:
                continue

            for forecast_start in date_generator(data_start_date + relativedelta(years=1), data_end_date):
                # try to eliminate some autocorrelation by 
                # keeping only the 1st and 15th as starting dates
                if forecast_start.day != 1 and forecast_start.day != 15:
                    continue
            
                cur_features = OrderedDict()
                cur_features["cell_id"] = cellid
                cur_features["forecast_start"] = forecast_start.isoformat()
                
                def store_outcomes(outcome_dict, time_window):
                    for category in outcome_categories:
                        cur_features["outcome_num_crimes_{0}_{1}".format(time_window, category)] = outcome_dict[category]
                
                # outcomes
                for day_window in DAY_WINDOWS:
                    edate = forecast_start + datetime.timedelta(days=day_window)
                    cur_outcomes = crimes_in_window(cellid, forecast_start, edate, outcome_index)
                    store_outcomes(cur_outcomes, "{0}days".format(day_window))
                
                def store_count_features(count_dict, time_window):
                    for category in crime_categories:
                        cur_features["p_num_crimes_{0}_{1}".format(time_window, category)] = count_dict[category]
                
                for month_num in range(1, 13):
                    cur_features["p_start_month_{0}".format(month_num)] = int(month_num == forecast_start.month)
                
                # daily counts
                for delta_days in range(3, 8):
                    ddate = forecast_start + datetime.timedelta(days=-delta_days)
                    counts_by_category = crimes_in_window(cellid, ddate, ddate, cell_count_index)
                    store_count_features(counts_by_category, "{0}days_ago".format(delta_days))
                
                # weekly counts
                for delta_weeks in range(4):
                    window_start = forecast_start + datetime.timedelta(days=-(delta_weeks * 7 + 9))
                    window_end = forecast_start + datetime.timedelta(days=-(delta_weeks * 7 + 3))
                    counts_by_category = crimes_in_window(cellid, window_start, window_end, cell_count_index)
                    store_count_features(counts_by_category, "{0}weeks_ago".format(delta_weeks+1))
                    
                # monthly counts
                for delta_months in range(4):
                    window_start = forecast_start + datetime.timedelta(days=-(delta_months * 30 + 32))
                    window_end = forecast_start + datetime.timedelta(days=-(delta_months * 30 + 3))
                    counts_by_category = crimes_in_window(cellid, window_start, window_end, cell_count_index)
                    store_count_features(counts_by_category, "{0}months_ago".format(delta_months+1))
                    
                # last year's values  
                py_start_date = forecast_start - relativedelta(years=1)
                for day_window in DAY_WINDOWS:
                    py_end = py_start_date + datetime.timedelta(days=6)
                    py_outcomes = crimes_in_window(cellid, py_start_date, py_end, cell_count_index)
                    store_count_features(py_outcomes, "py{0}days".format(day_window))
                
                # day structure
                for day_window in DAY_WINDOWS:
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
                # https://calendar.travelportland.com/calendar.xml
                if not writer:
                    writer = csv.DictWriter(handle, fieldnames=cur_features.keys())
                    writer.writeheader()
                writer.writerow(cur_features)

