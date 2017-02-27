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
import argparse
import numpy as np

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
    crime_categories = set(["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "OTHER"])
    outcome_categories = set(["BURGLARY", "MOTOR VEHICLE THEFT", "STREET CRIMES", "ALL"])

    cell_ids = set()
    total_crimes_by_category = defaultdict(int)
    crimes_by_cell_category = dict()
    with open("../../models/cells/cells-dim-{0}-meta.csv".format(cell_dimension), "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cell_id = row["id"]
            cell_ids.add(cell_id)

            crimes_by_cell_category[cell_id] = dict()
            crimes_by_cell_category[cell_id]["ALL"] = int(row["num.crimes"])
            crimes_by_cell_category[cell_id]["BURGLARY"] = int(row["num.crimes.BURGLARY"])
            crimes_by_cell_category[cell_id]["MOTOR VEHICLE THEFT"] = int(row["num.crimes.MOTOR.VEHICLE.THEFT"])
            crimes_by_cell_category[cell_id]["STREET CRIMES"] = int(row["num.crimes.STREET.CRIMES"])

            for category, count in crimes_by_cell_category[cell_id].items():
                total_crimes_by_category[category] += count

    mean_crimes_by_category = dict()
    for category, total in total_crimes_by_category.items():
        mean_crimes_by_category[category] = total / len(cell_ids)

    # mapping cell id and outcome category to pct of total crimes occurring in this cell
    cell_pct_crime = dict() 
    for cell_id, category_dict in crimes_by_cell_category.items():
        cell_pct_crime[cell_id] = dict()
        for category, count in category_dict.items():
            cell_pct_crime[cell_id][category] = (count + 1) / (mean_crimes_by_category[category] + 1)

    # mapping iso-formatted date to weather info
    weather_by_date = dict()
    with open("../../features/weather-features.csv", "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            weather_by_date[row["date"]] = row
    
    labor_stats_by_date = dict()
    with open("../../features/bls-features.csv", "r") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labor_stats_by_date[row["date"]] = row
        
    # (cellid, date) -> (category) -> count
    cell_count_index = defaultdict(lambda: defaultdict(int))

    data_start_date = None
    data_end_date = None
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
    study_end_date = dparser.parse("2017-03-01").date()

    writer = None
    with gzip.open(feature_file, "w+") as handle:
        for cellid in sorted(cell_ids, key=lambda c: int(c)):

            if int(cellid) % njobs != job_id:
                continue

            if int(cellid) % (100 + job_id) == 0:
                print("J{0} {1}: Working on cell {2}".format(job_id, datetime.datetime.now().isoformat(), cellid))

            for forecast_start in date_generator(data_start_date + relativedelta(years=1), study_end_date):
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
                    if edate > data_end_date:
                        cur_outcomes = defaultdict(lambda: None)
                    else:
                        cur_outcomes = crimes_in_window(cellid, forecast_start, edate, outcome_index)
                    store_outcomes(cur_outcomes, "{0}days".format(day_window))

                
                def store_count_features(count_dict, time_window):
                    for category in crime_categories:
                        cur_features["p_num_crimes_{0}_{1}".format(time_window, category)] = count_dict[category]
                
                for month_num in range(1, 13):
                    cur_features["p_start_month_{0}".format(month_num)] = int(month_num == forecast_start.month)

                # percentage of crime in each category
                for category, pct in cell_pct_crime[cellid].items():
                    cur_features["p_pct_crime_{0}".format(category)] = pct
                
                # St. Patrick's day
                for day_window in DAY_WINDOWS:
                    forecast_end_date = forecast_start + datetime.timedelta(days=day_window)
                    forecast_end_year = forecast_end_date.year
                    sp_day = datetime.date(forecast_end_year, 3, 17)
                    cur_features["p_sp_within_{0}days".format(day_window)] = int(sp_day >= forecast_start and sp_day <= forecast_end_date)

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
                        if d.weekday() == 1:
                            num_tuesdays += 1
                    cur_features["p_num_saturdays_next{0}days".format(day_window)] = num_saturdays
                    cur_features["p_num_fridays_next{0}days".format(day_window)] = num_fridays
                    cur_features["p_num_thursdays_next{0}days".format(day_window)] = num_thursdays
                    cur_features["p_num_tuesdays_next{0}days".format(day_window)] = num_tuesdays
                    
                # weather:
                for day_delta in [6, -7, -14, -30]:
                    window_start, window_end = sorted([forecast_start, 
                        forecast_start + datetime.timedelta(days=day_delta)])

                    def agg_weather_stat(attname, aggfun, prefix):
                        vals = []
                        for d in date_generator(window_start, window_end):
                            str_d = d.isoformat()
                            vals.append(float(weather_by_date[str_d][attname]))

                        cur_features["p_{0}_{1}_{2}{3}".format(
                                prefix, attname, 
                                "f" if day_delta > 0 else "h", abs(day_delta))] = aggfun(vals)

                    agg_weather_stat("precip_intensity", sum, "total")
                    agg_weather_stat("precip_intensity", np.mean, "mean")
                    agg_weather_stat("precip_accumulation", sum, "total")
                    agg_weather_stat("precip_accumulation", np.mean, "mean")
                    agg_weather_stat("snow", sum, "num")
                    agg_weather_stat("rain", sum, "num")
                    agg_weather_stat("sunlight_hours", sum, "total")
                    agg_weather_stat("high_temp", np.mean, "mean")
                    agg_weather_stat("low_temp", np.mean, "mean")
                    agg_weather_stat("cloud_cover", np.mean, "mean")
                    agg_weather_stat("cloud_cover", sum, "total")

                # labor statistics
                cur_labor_stats = labor_stats_by_date[forecast_start.isoformat()]
                cur_features["p_cur_wages"] = cur_labor_stats["wages"]
                cur_features["p_cur_labor_force"] = cur_labor_stats["labor_force"]
                cur_features["p_cur_employment"] = cur_labor_stats["employment"]
                cur_features["p_cur_unemployment"] = cur_labor_stats["unemployment"]
                
                # TODO: events
                # https://calendar.travelportland.com/calendar.xml
                if not writer:
                    writer = csv.DictWriter(handle, fieldnames=cur_features.keys())
                    if job_id == 0:
                        writer.writeheader()
                writer.writerow(cur_features)

if __name__ == "__main__":
    main()
