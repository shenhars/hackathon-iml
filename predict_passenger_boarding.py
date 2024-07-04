from datetime import datetime

import numpy as np
import pandas as pd
from typing import Optional
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
    X["passengers_up"] = y

    df = X.drop_duplicates() #remove duplicates - TODO nan
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station','alternative', 'trip_id_unique'], axis=1, inplace=True) # remove irelevant columns

    df = set_categoriel_feature(df)
    df = bus_in_the_station(df)

    #Change the arrival time estimation column from TRUE/FALSE to 1/0
    df['arrival_is_estimated'] = df['arrival_is_estimated'].map({True: 1, False: 0})

    # Check if the station ID is valid (is integer)
    df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    df = df[df['station_id_valid']]
    df = df.drop(['station_id_valid'], axis=1)

    df.dropna()
    y = df["passengers_up"]
    df.drop(["passengers_up"], axis=1)
    return df, y


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

    df = df.drop(['is_valid', 'arrival_time', 'door_closing_time'], axis=1)
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


def fit_model(X: pd.DataFrame, y):
    pass
