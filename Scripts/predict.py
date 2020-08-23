# predict probabilities for counts, given a set of future features
# execute: python predict.py
# This file can be scheduled to run at any time.
# It will generate hourly predictions for the remaining hours in today and N-additional days:
# - current hour - if the job is running at 8:45, the first prediction will be for time between 8AM and 9AM
# - next hours - till the end of the day, so until 23:59
# - and next N-days (so for 2 extra days - 48 additional predictions)

# Import necessary libraries
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import requests
import os
import json


# Define constants to use
DATASETS_PATH = '../Datasets'
OBJECT_CLASSES = ['person', 'vehicle']
PICKLED_SCALER_FILE = 'final_scaler__%%OBJECT_NAME%%.pickle'
PICKLED_MODEL_FILE = 'final_model__%%OBJECT_NAME%%.pickle'
PREDICTIONS_FILE = 'final_predictions__%%OBJECT_NAME%%.parquet'
N_SAMPLES = 2500  # how many simulations to execute in a Poisson process
PROBA_THRESH = 0.05  # reject probabilities below this threshold

# Define weather API parameters
API_BASE_URL = 'https://api.darksky.net/forecast'
API_KEY = '3d8137be1b3eb40d88ba1793e47f7071'
LAT, LONG = 51.802931199999996, -8.302591999999999  # camera coordinates
API_HEADERS = {'Accept-Encoding': 'gzip'}

# Define attributes to pull from the API
cur_keys = list(map(str.strip, """summary, precipIntensity, precipProbability, temperature, apparentTemperature,
humidity, windSpeed, windGust, windBearing, cloudCover, uvIndex, visibility""".split(',')))
daily_keys = list(map(str.strip, """summary, sunriseTime, sunsetTime, temperatureHigh, temperatureLow""".split(',')))


def make_ts(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, second).timestamp())


# pull most recent weather data for all hours
def make_api_url(ts: int) -> str:
    return f'{API_BASE_URL}/{API_KEY}/{LAT},{LONG},{ts}?exclude=hourly,flags,minutely&units=ca'


def predict(n_days: int = 3) -> None:
    """
    Create predictions for the next N days (including today)
    (meaning hours remaining for today + extra 2 days)
    :param n_days: int
    :return: None
    """

    # start date will be 30 minutes after current hour
    start_date = datetime.now().replace(minute=30, second=0)

    # end date will be + N-days at 23:45
    end_date = (start_date + timedelta(days=n_days-1)).replace(hour=23)

    # generate hourly date ranges
    idx = pd.date_range(start=start_date, end=end_date, freq='1H')
    hourly_df = pd.DataFrame({'dt': idx})

    # add date-time related features
    hourly_df['date'] = hourly_df['dt'].astype(str).str[0:19]
    hourly_df['hour'] = hourly_df['dt'].dt.hour
    hourly_df['n_month'] = hourly_df['dt'].dt.month
    hourly_df['day_of_week'] = hourly_df['dt'].dt.dayofweek
    hourly_df['is_weekend_day'] = (hourly_df['dt'].dt.dayofweek // 5 == 1).astype(int)

    # check if cached file with weather already exists, and use it
    # (to avoid making expensive API calls)
    cached_filename = f'{DATASETS_PATH}/weather_cached_{str(start_date)[:16]}_{str(end_date)[:16]}.csv'
    if os.path.isfile(cached_filename):
        logging.info(f'Using cached weather file {cached_filename}')
        weather_df = pd.read_csv(cached_filename)
    else:
        logging.info(f'Pulling weather data from API')

        # pull weather data for each hour from API
        dark_sky = []
        prev_cur_obj = {}
        prev_daily_obj = {}

        for index, d in tqdm(hourly_df.iterrows(), total=hourly_df.shape[0]):

            # extract date-time info from dt object
            args = (d['dt'].year, d['dt'].month, d['dt'].day, d['dt'].hour, d['dt'].minute)
            ts = make_ts(*args)

            # make a call to Dark Sky API
            url = make_api_url(ts)
            sky_data = requests.get(url, headers=API_HEADERS)
            if sky_data.status_code != 200:
                logging.error(f'Status code {sky_data.status_code} for url {url}')
                exit(1)

            sky_data = sky_data.json()
            currently = sky_data['currently']
            daily = sky_data['daily']

            # init object to save
            cur_obj = {'dt': d['dt']}

            # keep only selected data elements
            for k in cur_keys:
                if k in currently:
                    cur_obj[f'cur__{k}'] = currently[k]
                    prev_cur_obj[k] = currently[k]
                else:
                    cur_obj[f'cur__{k}'] = prev_cur_obj[k]
            for k in daily_keys:
                if k in daily['data'][0]:
                    cur_obj[f'daily__{k}'] = daily['data'][0][k]
                    prev_daily_obj[k] = daily['data'][0][k]
                else:
                    cur_obj[f'daily__{k}'] = prev_daily_obj[k]
            dark_sky.append(cur_obj)

        logging.info(f'Caching weather data')
        weather_df = pd.DataFrame(dark_sky)
        weather_df.to_csv(cached_filename, index=False)

    # add date-time related features, so we can merge the datasets together
    weather_df['dt'] = pd.to_datetime(weather_df['dt'])
    weather_df['date'] = weather_df['dt'].astype(str).str[0:19]

    # join detections and weather data
    logging.info(f'Merging hourly and weather datasets')
    merged = hourly_df.merge(weather_df, on=['date'])
    assert hourly_df.shape[0] == merged.shape[0]

    # generate dataset for prediction
    logging.info(f'Preparing features for predictions')
    use_cols = ['hour', 'n_month', 'day_of_week', 'is_weekend_day', 'cur__precipIntensity',
                'cur__precipProbability', 'cur__apparentTemperature', 'cur__humidity', 'cur__windSpeed',
                'cur__uvIndex']
    X = merged[use_cols]

    # load models for object classes and generate predictions
    for ob_class in OBJECT_CLASSES:
        logging.info(f'Loading scaler for {ob_class}')
        with open(f'{DATASETS_PATH}/{PICKLED_SCALER_FILE.replace("%%OBJECT_NAME%%", ob_class)}', 'rb') as f:
            scaler = pickle.load(f)

        # scale features using pickled scaler
        logging.info(f'Scaling features for {ob_class}')
        X_scaled = scaler.transform(X)

        logging.info(f'Loading model for {ob_class}')
        with open(f'{DATASETS_PATH}/{PICKLED_MODEL_FILE.replace("%%OBJECT_NAME%%", ob_class)}', 'rb') as f:
            model = pickle.load(f)

        logging.info(f'Generating predictions for {ob_class}')
        predictions = model.predict(X_scaled)

        logging.info(f'Generating probabilities from predictions for {ob_class}')
        predictions_probas = []
        expected_counts = []

        # sample from Poisson probability distribution for each prediction,
        # the assumption is that predictions can be interpreted as the rates
        # in a Poisson process (https://en.wikipedia.org/wiki/Poisson_distribution)
        for event_rate in predictions:
            samples = np.random.poisson(lam=event_rate, size=N_SAMPLES)

            # count unique counts from the simulation
            numbers, counts = np.unique(samples, return_counts=True)

            # generate probabilities from the unique counts
            probabilities = counts / counts.sum()

            # index with highest value will represent the count to expect
            expected_count = np.argmax(probabilities)
            expected_counts.append(expected_count)

            # filter out weak probabilities
            numbers_filtered = numbers[probabilities > PROBA_THRESH]
            probabilities_filtered = probabilities[probabilities > PROBA_THRESH]
            predictions_probas.append(json.dumps({
                'counts': numbers_filtered.tolist(),
                'probas': probabilities_filtered.tolist(),
            }))

        # create a copy of X with added predictions
        X_cp = X.copy()
        X_cp['pred'] = predictions
        X_cp['expected_count'] = expected_counts
        X_cp['pred_proba'] = predictions_probas

        # export to parquet (for now), ideally this should be persisted
        # in the DB, so the results can be analyzed later on,
        # parquet is a good choice, as it keeps the data types,
        # and pred_proba column contains Python dictionaries
        logging.info(f'Saving probabilities for {ob_class}')
        X_cp.to_parquet(f'{DATASETS_PATH}/{PREDICTIONS_FILE.replace("%%OBJECT_NAME%%", ob_class)}',
                        index=False)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s.%(msecs)03f %(levelname)s %(message)s",
                        level=logging.DEBUG, datefmt="%H:%M:%S")
    predict()
