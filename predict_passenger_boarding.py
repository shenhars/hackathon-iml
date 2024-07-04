from typing import Optional
from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def _preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None, is_train: bool = True):
    """
    preprocess the data
    """
    df = X.drop_duplicates() #remove duplicates
    df = split_into_areas(df)
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station','alternative',
             'trip_id_unique'], axis=1, inplace=True)  # remove irelevant columns

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = set_categoriel_feature(df)
    df = bus_in_the_station(df)
    df = get_trip_duration(df)

    # Check if the station ID is valid (is integer)
    df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    df = df[df['station_id_valid']]
    df = df.drop(['station_id_valid', 'arrival_time'], axis=1)

    df.dropna()
    y = y.loc[df.index]
    # feature_evaluation(df, y)
    return df, y

def split_into_areas(df):
    num_bins = 17

    # Create bins for longitude and latitude using linspace
    longitude_bins = np.linspace(df['longitude'].min(), df['longitude'].max(), num_bins)
    latitude_bins = np.linspace(df['latitude'].min(), df['latitude'].max(), num_bins)

    # Assign each point to a bin
    df['longitude_bin'] = np.digitize(df['longitude'], bins=longitude_bins, right=True)
    df['latitude_bin'] = np.digitize(df['latitude'], bins=latitude_bins, right=True)

    # Combine longitude and latitude bin values to create a unique area identifier
    df['area'] = df['longitude_bin'].astype(str) + '_' + df['latitude_bin'].astype(str)

    # Convert categorical area features to one-hot encoding
    df = pd.concat([df, pd.get_dummies(df['area'], prefix='area')], axis=1)

    # Drop the original bin columns and the combined identifier
    df = df.drop(['longitude_bin', 'latitude_bin', 'area'],axis = 1)

    return df


def get_trip_duration(df: pd.DataFrame):
    # Group by trip_id and aggregate
    grouped = df.groupby('trip_id').agg(
        min_station_index=('station_index', 'min'),
        max_station_index=('station_index', 'max'),
        min_arrival_time=('arrival_time', 'min'),
        max_arrival_time=('arrival_time', 'max')
    )

    # Calculate time difference
    grouped['trip_time'] = grouped['max_arrival_time'] - grouped['min_arrival_time']
    grouped['trip_time'] = grouped['trip_time'].dt.total_seconds()
    df = pd.merge(df, grouped[['trip_time']], left_on='trip_id', right_index=True, how='left')
    return df


def bus_in_the_station(df: pd.DataFrame):
    # validation of time'direction'
    # Convert time columns to datetime
    df = df.loc[df["arrival_time"].dropna().index]
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')

    # if the close time is before the arrival time- remove
    df['is_valid'] = df.apply(
        lambda row: row['door_closing_time'] > row['arrival_time'] if pd.notna(row['door_closing_time']) else False,
        axis=1)
    df = df[df['is_valid']]

    # duration that the door was opend
    df['door_duration'] = df.apply(
        lambda row: row['door_closing_time'] - row['arrival_time'] if pd.notna(row['door_closing_time']) else 0,
        axis=1)
    df['door_duration'] = df['door_duration'].dt.total_seconds()

    df = df.drop(['is_valid', 'door_closing_time'], axis=1)
    return df


def set_categoriel_feature(df: pd.DataFrame):
    directions = pd.get_dummies(df['direction'], prefix='direction')
    df = pd.concat([df, directions], axis=1)

    clusters = pd.get_dummies(df['cluster'], prefix='cluster')
    df = pd.concat([df, clusters], axis=1)

    df['arrival_hour'] = df['arrival_time'].dt.hour
    arrival_hour_dummies = pd.get_dummies(df['arrival_hour'], prefix='hour')
    df = pd.concat([df, arrival_hour_dummies], axis=1)

    df.drop(['part', 'direction', 'cluster', 'arrival_hour'], axis=1, inplace=True)

    boolean_cols = df.select_dtypes(include=['bool']).columns
    df[boolean_cols] = df[boolean_cols].astype(int)
    return df


def preprocess_test(df: pd.DataFrame):
    df = df.drop_duplicates()  # remove duplicates
    irrelevant_columns = ['latitude', 'longitude', 'station_name', 'trip_id_unique_station', 'alternative',
                          'trip_id_unique', 'part']
    df.drop(irrelevant_columns, axis=1, inplace=True)  # remove irelevant columns

    df = df.loc[df["arrival_time"].dropna().index]
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = set_categoriel_feature(df)
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')

    # duration that the door was opend
    df['door_duration'] = df.apply(
        lambda row: row['door_closing_time'] - row['arrival_time'] if pd.notna(row['door_closing_time']) else 0,
        axis=1)
    df['door_duration'] = df['door_duration'].dt.total_seconds()

    df = get_trip_duration(df)
    df = df.drop(['is_valid', 'door_closing_time'], axis=1)
    df.dropna()
    return df


def feature_evaluation(X: pd.DataFrame, y):
    for feature in X:
        covariance = np.cov(X[feature], y)[0, 1]
        std = (np.std(X[feature])*np.std(y))
        correlation = 0
        if std != 0:
            correlation = covariance / std

        plt.figure()
        plt.scatter(X[feature], y, color='blue', label=f'{feature} Values', s=1)
        plt.title(f'Correlation Between {feature} Values and Response\nPearson Correlation: {correlation}')
        plt.xlabel(f'{feature} Values')
        plt.ylabel('Response Values')

        # plt.show()


def load_data(path, encoding='ISO-8859-8'):
    return pd.read_csv(path, encoding=encoding)


def preprocess_data(X, y=None, is_train=True):
    if is_train:
        return _preprocess_data(X, y)
    return _preprocess_data(X, is_train=False)


def plot_predictions(y_true, y_pred, k, mse):
    plt.plot(y_true, color='red', label='Real Values')
    plt.plot(y_pred, color='blue', label='Predicted Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f"Real Values vs Predicted Values with k = {k} (MSE = {mse})")
    plt.legend()
    plt.show()


def train_and_evaluate(X_train, X_valid, y_train, y_valid, model_type, poly=None):
    if model_type == 'base':
        model = LinearRegression()
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


    X_train_processed = X_train
    X_valid_processed = X_valid

    logging.info("training...")
    model.fit(X_train_processed, y_train)
    # y_pred_on_valid = model.predict(X_valid_processed)
    y_pred_on_valid = model.predict(X_valid_processed)
    mse = mean_squared_error(y_pred_on_valid, y_valid)
    mse = round(mse, 3)

    return model, mse, y_pred_on_valid


def main():
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    parser.add_argument('--train', type=bool, required=True, help="is training or test")
    parser.add_argument('--model_type', type=str, required=True, choices=['base', 'rf', 'gb'],
                        help="type of model to use")
    parser.add_argument('--bootstrap', type=bool, required=True, help="bootstrap")

    args = parser.parse_args()

    is_train = args.train
    seed = 42
    test_size = 0.2

    print("train" if is_train else "test")

    if is_train:
        for k in range(3, 4):
            # poly = PolynomialFeatures(degree=k, include_bias=False)

            df = load_data(args.training_set)
            X, y = df.drop("passengers_up", axis=1), df.passengers_up

            preprocess_x, preprocess_y = preprocess_data(X, y)
            logging.info("preprocessing train...")
            X_train, X_valid, y_train, y_valid = train_test_split(preprocess_x, preprocess_y, test_size=test_size,
                                                                  random_state=seed)
            X_train, X_valid, y_train, y_valid = np.array(X_train), np.array(X_valid), np.array(y_train), np.array(
                y_valid)

            model, mse, y_pred_on_valid = train_and_evaluate(X_train, X_valid, y_train, y_valid, args.model_type)
            print(f"k={k}, mse={mse}")
            plot_predictions(y_valid, y_pred_on_valid, k, mse)

            # save the model
            with open(f"model_task1.sav", "wb") as f:
                pickle.dump(model, f)

    else:
        with open("model_task1.sav", "rb") as file:
            model = pickle.load(file)

            df = load_data(args.test_set)
            logging.info("preprocessing test...")
            X_test_processed = preprocess_data(df, is_train=False)

            logging.info("predicting...")
            y_pred = model.predict(X_test_processed)

            logging.info(f"predictions saved to {args.out}")
            predictions = pd.DataFrame(
                {'trip_id_unique_station': X_test_processed['trip_id_unique_station'], 'passenger_up': y_pred})
            predictions.to_csv(args.out, index=False)