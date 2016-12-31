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
    for cur_date in date_generator(dparser.parse(args.start_date), dparser.parse(args.end_date)):
        print("Downloading data for {0}".format(cur_date.isoformat()))
        req_url = "https://api.darksky.net/forecast/{0}/45.522919,-122.671046,{1}?exclude=currently,minutely,hourly".format(api_key, cur_date.isoformat())
        resp = requests.get(req_url)
        full_response = resp.json()
        print(full_response)
        
main()