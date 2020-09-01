# train models for selected object classes
# execute: python train.py

# Import necessary libraries
import pandas as pd
import numpy as np
import logging
import warnings

# Gradient Boosting Decision Tree in sklearn produces warnings, so suppress those
warnings.simplefilter('ignore')
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.simplefilter('always')

from sklearn.metrics import (mean_squared_error, r2_score, mean_poisson_deviance,
                             mean_absolute_error, accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
import pickle

# Define constants to use
DATASETS_PATH = '../Datasets'
DETECTIONS_FILENAME = 'UniqueObjectDetections__%%OBJECT_NAME%%__2019-09-09_2020-03-02.csv'
WEATHER_FILENAME = 'dark_sky_data_2019-09-09_2020-03-02.csv'
OBJECT_CLASSES = ['person', 'vehicle']
DOWNTIME_DATES = ['2020-01-13', '2020-01-14', '2020-02-28']
PICKLED_SCALER_FILE = 'final_scaler__%%OBJECT_NAME%%.pickle'
PICKLED_MODEL_FILE = 'final_model__%%OBJECT_NAME%%.pickle'
RUN_CROSS_VAL = True


def train() -> None:
    """
    Create a model for each object class, currently using Gradient Boosting
    Decision Tree from sklearn, as it is accurate, fairly robust and fast
    to train (only a few seconds).
    :return: None
    """
    # load weather data
    weather = pd.read_csv(f'{DATASETS_PATH}/{WEATHER_FILENAME}')

    # add date-time related features, so we can merge the datasets later
    weather['dt'] = pd.to_datetime(weather['dt'])
    weather['date'] = weather['dt'].dt.date
    weather['date'] = pd.to_datetime(weather['date'])
    weather['hour'] = weather['dt'].dt.hour
    
    # load datasets for object classes
    for ob_class in OBJECT_CLASSES:

        # load dataset for an object class
        df = pd.read_csv(f'{DATASETS_PATH}/{DETECTIONS_FILENAME.replace("%%OBJECT_NAME%%", ob_class)}')
        logging.debug(f'Initial shape for {ob_class}: {df.shape}')

        # add date time fields
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['date'] = pd.to_datetime(df['date'])

        # use Pandas handy resample feature to fill in gaps with 0's
        object_detections = df.set_index('date_time').resample('H')['dummy_var'].sum().reset_index()
        object_detections.columns = ['date_time', 'obs_count']
        object_detections['date'] = object_detections['date_time'].dt.date.astype(str)
        object_detections['hour'] = object_detections['date_time'].dt.hour
        object_detections = object_detections[['date', 'hour', 'obs_count']]
        logging.debug(f'Resampled shape for {ob_class}:{df.shape}')

        # remove any entries where we know that there was an error in measurements
        orig_size = object_detections.shape[0]
        idx = object_detections['date'].isin(DOWNTIME_DATES)
        object_detections = object_detections.loc[~idx]
        logging.debug(f'Removed {orig_size - object_detections.shape[0]} records')

        # add date-time related features
        object_detections['date'] = pd.to_datetime(object_detections['date'])
        object_detections['n_month'] = object_detections['date'].dt.month
        object_detections['n_week_in_month'] = (object_detections['date'].dt.day - 1) // 7 + 1
        object_detections['day_of_week'] = object_detections['date'].dt.dayofweek
        object_detections['day_of_week_name'] = object_detections['date'].dt.day_name()
        object_detections['is_weekend_day'] = (object_detections['date'].dt.dayofweek // 5 == 1).astype(int)
        object_detections['day_of_week_name_short'] = 'WeekDay'
        idx = object_detections['day_of_week_name'].isin(['Saturday', 'Sunday'])
        object_detections.loc[idx, 'day_of_week_name_short'] = object_detections[idx]['day_of_week_name']

        # join detections and weather data
        merged = object_detections.merge(weather, on=['date', 'hour'])

        # define features to use
        use_cols = ['hour', 'n_month', 'day_of_week', 'is_weekend_day', 'cur__precipIntensity',
                    'cur__precipProbability', 'cur__apparentTemperature', 'cur__humidity', 'cur__windSpeed',
                    'cur__uvIndex']

        # define x and y
        X = merged[use_cols]
        y = merged['obs_count']

        # check if KFold Cross Validation is required
        if RUN_CROSS_VAL:

            # run a cross validation and display results
            logging.info('Running cross validation')

            kf = KFold(n_splits=3, shuffle=True)
            # Iterate over each train-test split and calculate scores
            mpds = []
            mses = []
            maes = []
            r2s = []
            acc = []
            for train_index, test_index in tqdm(kf.split(X)):
                # Split train-test
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                # Scale train and test sets
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                # Train model
                model = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.01, max_iter=550,
                                                      max_bins=52, l2_regularization=0.1, verbose=0,
                                                      validation_fraction=0.1, n_iter_no_change=5,
                                                      early_stopping=True)
                model.fit(X_train_scaled, y_train)
                # Make predictions and round them off
                y_pred = model.predict(X_test_scaled)
                y_pred_rounded = np.array([round(p) for p in y_pred])
                # Calculate scores
                mpds.append(mean_poisson_deviance(y_test, y_pred + 0.00001))
                mses.append(mean_squared_error(y_test, y_pred))
                maes.append(mean_absolute_error(y_test, y_pred))
                r2s.append(r2_score(y_test, y_pred))
                acc.append(accuracy_score(y_test, y_pred_rounded))

            # log stats (ideally this should be persisted)
            logging.info(f'mean mpd for {ob_class}:, {np.mean(mpds)}')
            logging.info(f'mean mse for {ob_class}:, {np.mean(mses)}')
            logging.info(f'mean mae for {ob_class}:, {np.mean(maes)}')
            logging.info(f'mean r2 for {ob_class}:, {np.mean(r2s)}')
            logging.info(f'mean acc for {ob_class}:, {np.mean(acc)}')

        # now scale whole dataset (without train/test split)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # train model using scaled, whole dataset
        logging.info('Training model on all data')
        model = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.01, max_iter=550,
                                              max_bins=52, l2_regularization=0.1, verbose=0,
                                              validation_fraction=0.1, n_iter_no_change=5,
                                              early_stopping=True)
        model.fit(X_scaled, y)

        # save scaler as pickle
        logging.info(f'Saving scaler for {ob_class}')
        scaler_filename = f'{DATASETS_PATH}/{PICKLED_SCALER_FILE.replace("%%OBJECT_NAME%%", ob_class)}'
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)

        # save model as pickle
        logging.info(f'Saving model for {ob_class}')
        model_filename = f'{DATASETS_PATH}/{PICKLED_MODEL_FILE.replace("%%OBJECT_NAME%%", ob_class)}'
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s.%(msecs)03f %(levelname)s %(message)s",
                        level=logging.DEBUG, datefmt="%H:%M:%S")
    train()
