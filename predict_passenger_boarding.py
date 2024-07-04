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
    df.drop(['latitude', 'longitude', 'station_name', 'cluster', 'alternative', 'part'], axis=1, inplace=True) # remove irelevant columns
    directions = pd.get_dummies(df['direction'], prefix='direction_1') #add first direction
    df = pd.concat([df, directions], axis=1)
    df = df.drop('direction', axis=1, inplace=True)

    #validation of time
    #Convert time columns to datetime

    df = df['door_closing_time'].fillna(value="00:00:00") #fill missing values with 0 so it would be droped when calling dropna
    df = df['arrival_time'].fillna(value="00:00:00") #fill missing values with 0 so it would be droped when calling dropna

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')

    df['arrival_time_valid'] = df['arrival_time'].apply(is_valid_time)
    df['door_closing_time_valid'] = df['door_closing_time'].apply(lambda x: is_valid_time(x) if pd.notna(x) else False)

    df = df[df['door_closing_time_valid']]
    df = df[df['arrival_time_valid']]
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'])
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])


    #if the close time is before the arrival time- remove
    # df['is_valid'] = df.apply(
    #     lambda row: row['close_door_time'].dt.hour > row['arrival_time'].dt.hour if pd.notna(row['close_door_time']) else False, axis=1)
    # df = df[df['is_valid']]

    #open or not?
    df['was_opened'] = df['door_closing_time'].apply(lambda x: 1 if x > 0 else 0)

    #duration that the door was opend
    df['time_door_was_opened'] = df['door_closing_time'] - df['arrival_time']
    df = df.drop(['arrival_time', 'door_closing_time'], axis=1)


    #Change the arrival time estimation column from TRUE/FALSE to 1/0
    df['arrival_is_estimated'] = df['arrival_is_estimated'].map({True: 1, False: 0})

    # Check if the station ID is valid (is integer)
    # df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    # df = df[df['station_id_valid']]
    # df = df.drop(['station_id_valid'], axis=1)

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
