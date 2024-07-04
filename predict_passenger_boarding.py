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
    df.drop(['latitude', 'longitude', 'station_name', 'cluster', 'alternative', 'part', 'trip_id_unique_station', 'trip_id_unique'], axis=1, inplace=True) # remove irelevant columns
    directions = pd.get_dummies(df['direction'], prefix='direction_1') #add first direction
    df = pd.concat([df, directions], axis=1)
    df.drop('direction', axis=1, inplace=True)
    df['direction_1_1'] = df['direction_1_1'].map({True: 1, False: 0})
    df['direction_1_2'] = df['direction_1_2'].map({True: 1, False: 0})

    #validation of time
    #Convert time columns to datetime
    df = df.loc[df["arrival_time"].dropna().index]
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')

    #if the close time is before the arrival time- remove
    df['is_valid'] = df.apply(
        lambda row: row['door_closing_time'] > row['arrival_time'] if pd.notna(row['door_closing_time']) else False, axis=1)
    df = df[df['is_valid']]

    # duration that the door was opend
    df['door_duration'] = df.apply(
        lambda row: row['door_closing_time'] - row['arrival_time'] if pd.notna(row['door_closing_time']) else 0,
        axis=1)
    df['door_duration'] = df['door_duration'].dt.total_seconds()

    df = df.drop(['is_valid', 'arrival_time', 'door_closing_time'], axis=1)


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

def is_valid_time(time_str):
    try:
        datetime.strptime(time_str, '%H:%M:%S')
        return True
    except:
        return False

def fit_model(X: pd.DataFrame, y):
    pass
