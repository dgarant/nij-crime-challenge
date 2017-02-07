import requests
import requests_cache
import os
import sys
from dateutil import parser as dparser
from datetime import date, timedelta
import re
import scipy as sp
import time
from scipy import interpolate as spinterp
from collections import defaultdict
import csv

requests_cache.install_cache("bls_cache", allowable_methods=["GET", "POST"])

start_date = dparser.parse(sys.argv[1])
end_date = dparser.parse(sys.argv[2])

def date_generator(from_date, to_date):
    while from_date <= to_date:
        yield from_date
        from_date = from_date + timedelta(days=1)

adjustment_code = "U"
#area_code = "IM4138900000000"
area_code = "MT4138900000000"
csa_code = "CS440" # 'combined' statistical area

unemployment_series = "LA" + adjustment_code + area_code + "03"
labor_force_series = "LA" + adjustment_code + area_code + "04"
employment_series = "LA" + adjustment_code + area_code + "05"
wage_series = "EN" + adjustment_code + csa_code + "1" + "0" + "0" + "10"
api_key = os.environ["BLS_API_KEY"]

series_keys = {
        unemployment_series : "unemployment",
        labor_force_series : "labor_force",
        employment_series : "employment",
        wage_series : "wages"
    }

resp = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", 
    json={"seriesid" : series_keys.keys(), "startyear" : "2012", "endyear" : "2017",
          "registrationKey" : api_key})
response = resp.json()

print("Status: {0} Messages:".format(response["status"]))
for message in response["message"]:
    print(message)

result_series = response["Results"]["series"]
print("Got {0} series".format(len(result_series)))
series_values = defaultdict(dict)
for series in result_series:
    sname = series_keys[series["seriesID"]]

    series_utimes = []
    series_vals = []
    for rec in series["data"]:
        mgroups = re.match("M([0-9]+)", rec["period"]).groups()
        month = int(mgroups[0])
        year = int(rec["year"])
        series_utimes.append(time.mktime(date(year, month, 1).timetuple()))
        series_vals.append(float(rec["value"]))

    interpfun = spinterp.interp1d(series_utimes, series_vals, kind="linear", fill_value="extrapolate")

    for d in date_generator(start_date, end_date):
        utime = time.mktime(d.timetuple())
        series_val = interpfun(utime)
        series_values[d.date()][sname] = series_val

with open("../../features/bls-features.csv", "w+") as handle:
    val_keys = series_keys.values()
    writer = csv.DictWriter(handle, fieldnames=["date"] + list(val_keys))
    writer.writeheader()
    for d, v in sorted(series_values.items()):
        row = {"date" : d.isoformat()}
        for key in val_keys:
            row[key] = v[key]
        writer.writerow(row)

