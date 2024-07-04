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

    X['door_closing_time'] = X['door_closing_time'].fillna(value=0) #fill missing values with 0 so it would be droped when calling dropna
    df = X.drop_duplicates() #remove duplicates - TODO nan
    df.drop(['latitude', 'longitude', 'station_name', 'cluster', 'alternative', 'part'], axis=1) # remove irelevant columns
    directions = pd.get_dummies(df['direction'], prefix='direction_1') #add first direction
    df = pd.concat([df, directions], axis=1)
    df.drop('direction', axis=1, inplace=True)

    # validation of time
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'],errors='coerce').notna()
    df['arrival_time'] = pd.to_datetime(df['arrival_time'],errors='coerce').notna()

    # if the close time is before the arrival time- remove
    df['is_valid'] = df.apply(
        lambda row: row['close_door_time'] > row['arrival_time'] if pd.notna(row['close_door_time']) else False, axis=1)
    df = df[df['is_valid'] == True]
    df['door_closing_time'] = df['door_closing_time'].fillna(value=0) #fill missing values with 0 so it would be droped when calling dropna

    # open or not?
    df['was_opened'] = df['door_closing_time'].apply(lambda x: 1 if x > 0 else 0)

    #duration that the door was opend
    df['time_door_was_opened'] = df['door_closing_time'] - df['arrival_time']
    df.drop(["is_valid", 'arrival_time', 'door_closing_time'], axis=1)

    # Change the arrival time estimation column from TRUE/FALSE to 1/0
    df['arrival_is_estimated'] = df['arrival_is_estimated'].map({True: 1, False: 0})

    # Check if the station ID is valid (is integer)
    df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    df['station_id'] = df[df['station_id_valid']]
    df.drop(['station_id_valid'], axis=1)

    df.dropna()


def fit_model(X: pd.DataFrame, y):
    pass
