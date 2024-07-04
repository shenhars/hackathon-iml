import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
from pprint import pprint


def preprosses_united_data(X: )


def _preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None, is_train: bool = True):
    """
    preprocess the data
    """
    # combine x and y
    X["passengers_up"] = y

    X['door_closing_time'] = X['door_closing_time'].fillna(value=0) #fill missing values with 0 so it would be droped when calling dropna
    df = X.dropna().drop_duplicates() #remove nan and duplicates
    df.dropna('latitude', 'longitude', 'station_name', axis=1) # remove irelevant columns
    directions = pd.get_dummies(df['direction'], prefix='direction_1') #add first direction
    df = pd.concat([df, directions], axis=1)
    df.drop('direction', axis=1, inplace=True)

    #calculate the time that the door bus is open in every stop
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'])
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])
    df['was_opened'] = df['door_closing_time'].apply(lambda x: 1 if x > 0 else 0)
    df['time_door_was_opened'] = df['door_closing_time'] - df['arrival_time']
    df.loc[df['was_opened'] == 0, 'years_since_renovation'] = df['time_door_was_opened']


def fit_model(X: pd.DataFrame, y):
    pass
