
import pandas as pd
from typing import Optional


def _preprocess_data(X: pd.DataFrame, is_train: bool = True):
    """
    preprocess the data
    """
    df = X.drop_duplicates()
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station','alternative',
             'trip_id_unique', 'part'], axis=1, inplace=True)  # remove irelevant columns

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = set_categoriel_feature(df)
    df = bus_in_the_station(df)
    df = get_trip_duration(df)

    # Check if the station ID is valid (is integer)
    df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    df = df[df['station_id_valid']]
    df = df.drop(['station_id_valid', 'arrival_time'], axis=1)

    df.dropna()
    y = df['trip_time']
    df = df.drop(["trip_time"], axis=1)
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

    df.drop(['direction', 'cluster', 'arrival_hour'], axis=1, inplace=True)

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

    df = df.drop(['is_valid', 'door_closing_time', 'arrival_time'], axis=1)
    df.dropna()
    return df
