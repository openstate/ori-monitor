#!/usr/bin/env python

import sys
import csv
import json
from time import sleep
import datetime
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

from settings import ORI_API_URL


def get_governments():
    resp = requests.post(
        '%s/search/organizations' % (ORI_API_URL,),
        data=json.dumps({
            'facets': {
                'start_date': {}
            },
            'filters': {
                'classification': {
                    'terms': ['Municipality']  # province also
                }
            },
            'size': 10
        }))
    return resp.json()


def get_government_slug(g):
    return g['meta']['collection']


def get_dates_for_government(g):
    resp = requests.post(
        '%s/%s/events/search' % (ORI_API_URL, get_government_slug(g),),
        data=json.dumps({
            'facets': {
                'start_date': {
                    'interval': 'day'
                }
            },
            'size': 0
        }))
    dates = {
        x['key_as_string'][0:10]: x['doc_count'] for x in
        resp.json()['facets']['start_date']['buckets']}
    return dates


def output_dates(g, dates):
    f = open('%s.csv' % (get_government_slug(g),), 'w')

    with f:

        writer = csv.writer(f)
        writer.writerow(['ds', 'y'])

        for dt, c in dates.items():
            writer.writerow([dt, c])


def predict(g):
    dates_path = './%s.csv' % (get_government_slug(g),)
    dt_today = datetime.datetime.today().date()
    df = pd.read_csv(dates_path)
    # ensure our ds value is truly datetime
    df['ds'] = pd.to_datetime(df['ds'])
    # filtering here on >=1995, just to pull the last ~20 years of production
    # information
    start_date = str(dt_today - datetime.timedelta(days=365))
    mask = (df['ds'] > start_date)
    df = df.loc[mask]
    # initialize Prophet
    m = Prophet()
    # point towards dataframe
    m.fit(df)
    # set future prediction window of 2 years
    future = m.make_future_dataframe(periods=10)
    # preview our data -- note that Prophet is only showing future dates (not
    # values), as we need to call the prediction method still
    forecast = m.predict(future)
    start_date = str(dt_today)
    mask = (forecast['ds'] > start_date)
    forecast = forecast.loc[mask]

    datespredict_path = './prediction_%s.csv' % (get_government_slug(g),)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10).to_csv(
        datespredict_path)


def main(argv):
    for g in get_governments()['organizations']:
        print(get_government_slug(g))
        dates = get_dates_for_government(g)
        output_dates(g, dates)
        predict(g)
        sleep(1)
    return 0

if __name__ == '__main__':
        sys.exit(main(sys.argv))
