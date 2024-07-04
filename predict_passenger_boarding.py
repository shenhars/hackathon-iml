from datetime import datetime

import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
from pprint import pprint


def _preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None, is_train: bool = True):
    """
    preprocess the data
    """
    # combine x and y
    # X["passengers_up"] = y

    df = X.drop_duplicates() #remove duplicates
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station','alternative',
             'trip_id_unique'], axis=1, inplace=True)  # remove irelevant columns

    df = set_categoriel_feature(df)
    df = bus_in_the_station(df)
    df = get_trip_duration(df)

    # Check if the station ID is valid (is integer)
    df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    df = df[df['station_id_valid']]
    df = df.drop(['station_id_valid'], axis=1)

    df.dropna()
    df['arrival_time'] = (pd.to_datetime("00:00:00", format='%H:%M:%S') - df['arrival_time']).dt.total_seconds()
    y = y.loc[df.index]
    feature_evaluation(df, y)
    return df, y


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
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
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

    parts = pd.get_dummies(df['part'], prefix='part')
    df = pd.concat([df, parts], axis=1)

    df.drop(['part', 'direction', 'cluster'], axis=1, inplace=True)

    boolean_cols = df.select_dtypes(include=['bool']).columns
    df[boolean_cols] = df[boolean_cols].astype(int)
    return df


def preprocess_test(df: pd.DataFrame):

    df = df.drop_duplicates()  # remove duplicates
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station', 'alternative',
             'trip_id_unique'], axis=1, inplace=True)  # remove irelevant columns

    # df = get_trip_duration(df)
    df = set_categoriel_feature(df)
    df = df.loc[df["arrival_time"].dropna().index]
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')

    # duration that the door was opend
    df['door_duration'] = df.apply(
        lambda row: row['door_closing_time'] - row['arrival_time'] if pd.notna(row['door_closing_time']) else 0,
        axis=1)
    df['door_duration'] = df['door_duration'].dt.total_seconds()

    df = get_trip_duration(df)
    df['arrival_time'] = (pd.to_datetime("00:00:00", format='%H:%M:%S') - df['arrival_time']).dt.total_seconds()
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

        plt.show()
def fit_model(X: pd.DataFrame, y):
    pass
