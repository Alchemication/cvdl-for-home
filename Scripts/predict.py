# predict probabilities for counts, given a set of future features
# execute: python predict.py

# Import necessary libraries
import logging
import pickle
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import requests
import os


# Define constants to use
DATASETS_PATH = '../Datasets'
OBJECT_CLASSES = ['person', 'vehicle']
PICKLED_MODEL_FILE = 'final_model__%%OBJECT_NAME%%.pickle'

# Define weather API parameters
API_BASE_URL = 'https://api.darksky.net/forecast'
API_KEY = '3d8137be1b3eb40d88ba1793e47f7071'
LAT, LONG = 51.802931199999996, -8.302591999999999  # camera coordinates
API_HEADERS = {'Accept-Encoding': 'gzip'}

# Define attributes to pull from the API
cur_keys = list(map(str.strip, """summary, precipIntensity, precipProbability, temperature, apparentTemperature,
humidity, windSpeed, windGust, windBearing, cloudCover, uvIndex, visibility""".split(',')))
daily_keys = list(map(str.strip, """summary, sunriseTime, sunsetTime, temperatureHigh, temperatureLow""".split(',')))


def make_ts(year, month, day, hour, minute, second=0):
    return int(datetime(year, month, day, hour, minute, second).timestamp())


# pull most recent weather data for all hours
def make_api_url(ts):
    return f'{API_BASE_URL}/{API_KEY}/{LAT},{LONG},{ts}?exclude=hourly,flags,minutely&units=ca'


def predict(n_days: int = 3) -> None:
    """
    Create predictions for the next N days (including today)
    (meaning hours remaining for today + extra 2 days)
    :param n_days: int
    :return:
    """

    # fetch weather data

    # start date will be 45 minutes after current hour
    start_date = datetime.now().replace(minute=30, second=0)

    # end date will be + N-days at 23:45
    end_date = (start_date + timedelta(days=n_days-1)).replace(hour=23)

    # check if cached file with weather already exists, and use it
    # (to avoid making expensive API calls)
    cached_filename = f'{DATASETS_PATH}/weather_cached_{str(start_date)[:16]}_{str(end_date)[:16]}.csv'
    if os.path.isfile(cached_filename):
        logging.info(f'Using cached weather file {cached_filename}')
        weather_df = pd.read_csv(cached_filename)
    else:
        logging.info(f'Pulling weather data from API')

        # generate hourly date ranges
        idx = pd.date_range(start=start_date, end=end_date, freq='1H')
        df = pd.DataFrame({'dt': idx})

        # pull weather data for each hour from API
        dark_sky = []
        prev_cur_obj = {}
        prev_daily_obj = {}

        for index, d in tqdm(df.iterrows(), total=df.shape[0]):

            # extract date-time info from dt object
            args = (d.dt.year, d.dt.month, d.dt.day, d.dt.hour, d.dt.minute)
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
            cur_obj = {'ts': ts, 'dt': d}

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
        weather_df['dt'] = df['dt']
        weather_df.to_csv(cached_filename, index=False)

    # load models

    # generate dataset for prediction

    # predict

    # generate probabilities from predictions

    # persist latest predictions and probabilities
    pass


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s.%(msecs)03f %(levelname)s %(message)s",
                        level=logging.DEBUG, datefmt="%H:%M:%S")
    predict()
