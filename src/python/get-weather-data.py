import argparse
import requests_cache
import requests
import csv
import json
import os
from collections import defaultdict
from time import strptime
import datetime
import calendar
import sys
from dateutil import parser as dparser

requests_cache.install_cache("weather_cache")

def date_generator(from_date, to_date):
    while from_date <= to_date:
        yield from_date
        from_date = from_date + datetime.timedelta(days=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    args = parser.parse_args()

    api_key = os.environ["DARKSKY_API_KEY"]
    with open("../../features/weather-features.csv", "w+") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "precip_intensity", "precip_accumulation", 
            "snow", "rain", "sunlight_hours", 
            "high_temp", "low_temp", "cloud_cover"])
        writer.writeheader()

        for cur_date in date_generator(dparser.parse(args.start_date), dparser.parse(args.end_date)):
            print("Downloading data for {0}".format(cur_date.isoformat()))
            req_url = "https://api.darksky.net/forecast/{0}/45.522919,-122.671046,{1}?exclude=currently,minutely,hourly".format(api_key, cur_date.isoformat())
            resp = requests.get(req_url)
            full_response = resp.json()

            daily_data = full_response["daily"]["data"][0]
            cur_features = dict()
            cur_features["date"] = cur_date.strftime("%Y-%m-%d")
            cur_features["precip_intensity"] = daily_data["precipIntensity"]
            cur_features["precip_accumulation"] = daily_data.get("precipAccumulation", 0)
            cur_features["snow"] = int(daily_data.get("precipType", "") == "snow")
            cur_features["rain"] = int(daily_data.get("precipType", "") == "rain")

            cur_features["sunlight_hours"] = (daily_data["sunsetTime"] - daily_data["sunriseTime"]) / 3600.0
            cur_features["high_temp"] = daily_data["apparentTemperatureMax"]
            cur_features["low_temp"] = daily_data["apparentTemperatureMin"]
            cur_features["cloud_cover"] = daily_data.get("cloudCover", 0)

            writer.writerow(cur_features)

main()
