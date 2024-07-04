from typing import Optional
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler


def _preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None, is_train: bool = True):
    """
    preprocess the data
    """
    df = X.drop_duplicates()  # remove duplicates
    df = split_into_areas(df)
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station', 'alternative', 'trip_id_unique', 'part'], axis=1, inplace=True)  # remove irrelevant columns

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = set_categorical_features(df)
    df = bus_in_the_station(df)
    df = get_trip_duration(df)

    # Check if the station ID is valid (is integer)
    df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    df = df[df['station_id_valid']]
    df = df.drop(['station_id_valid', 'arrival_time'], axis=1)

    df.dropna(inplace=True)
    if y is not None:
        y = y.loc[df.index]
    return df, y

def get_trip_duration(df: pd.DataFrame):
    grouped = df.groupby('trip_id').agg(
        min_station_index=('station_index', 'min'),
        max_station_index=('station_index', 'max'),
        min_arrival_time=('arrival_time', 'min'),
        max_arrival_time=('arrival_time', 'max')
    )
    grouped['trip_time'] = (grouped['max_arrival_time'] - grouped['min_arrival_time']).dt.total_seconds()
    df = pd.merge(df, grouped[['trip_time']], left_on='trip_id', right_index=True, how='left')
    return df

def split_into_areas(df):
    # Standardize the longitude and latitude
    scaler = StandardScaler()
    df[['longitude_std', 'latitude_std']] = scaler.fit_transform(df[['longitude', 'latitude']])

    # Apply K-Means clustering
    num_clusters = 25
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['cluster_'] = kmeans.fit_predict(df[['longitude_std', 'latitude_std']])

    # Convert cluster labels to one-hot encoding
    df = pd.concat([df, pd.get_dummies(df['cluster_'], prefix='cluster')], axis=1)

    # Drop unnecessary columns
    df = df.drop(['longitude_std', 'latitude_std', 'cluster_'],axis = 1)

    return df

def bus_in_the_station(df: pd.DataFrame):
    df = df.loc[df["arrival_time"].dropna().index]
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')
    df['is_valid'] = df.apply(lambda row: row['door_closing_time'] > row['arrival_time'] if pd.notna(row['door_closing_time']) else False, axis=1)
    df = df[df['is_valid']]
    df['door_duration'] = (df['door_closing_time'] - df['arrival_time']).dt.total_seconds()
    df = df.drop(['is_valid', 'door_closing_time'], axis=1)
    return df

def set_categorical_features(df: pd.DataFrame):
    df = pd.get_dummies(df, columns=['direction', 'cluster'], prefix=['direction', 'cluster'])
    df['arrival_hour'] = df['arrival_time'].dt.hour
    df = pd.get_dummies(df, columns=['arrival_hour'], prefix='hour')
    return df

def preprocess_test(df: pd.DataFrame):
    df = split_into_areas(df)
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station', 'alternative', 'trip_id_unique', 'part'], axis=1, inplace=True)  # remove irrelevant columns

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = set_categorical_features(df)
    df = df.loc[df["arrival_time"].dropna().index]
    df['door_closing_time'] = pd.to_datetime(df['door_closing_time'], format='%H:%M:%S')
    df['door_duration'] = (df['door_closing_time'] - df['arrival_time']).dt.total_seconds()
    df = df.drop([ 'door_closing_time'], axis=1)
    df = get_trip_duration(df)
    df = df.drop(['arrival_time'], axis=1)

    return df

def load_data(path, encoding='ISO-8859-8'):
    return pd.read_csv(path, encoding=encoding)

def train_and_evaluate(X_train, X_valid, y_train, y_valid, model_type):
    if model_type == 'base':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    elif model_type == 'lgb':
        model = lgb.LGBMRegressor(random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    param_grid = {
        'base': {},
        'ridge': {'alpha': [0.1, 1.0, 10.0]},
        'rf': {'n_estimators': [100, 200]},
        'gb': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
        'xgb': {'n_estimators': [100, 200], 'learning_rate': [0.001, 0.01, 0.1]},
        'lgb': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
    }

    grid_search = GridSearchCV(model, param_grid[model_type], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_pred_on_valid = best_model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred_on_valid)
    mse = round(mse, 3)

    return best_model, mse, y_pred_on_valid


def plot_predictions(y_true, y_pred, k, mse):
    plt.plot(y_true, color='red', label='Real Values')
    plt.plot(y_pred, color='blue', label='Predicted Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f"Real Values vs Predicted Values with k = {k} (MSE = {mse})")
    plt.legend()
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True, help="path of the output file as required in the task description")
    parser.add_argument('--train', type=bool, required=True, help="is training or test")
    parser.add_argument('--model_type', type=str, required=True, choices=['base', 'ridge', 'rf', 'gb', 'xgb', 'lgb'], help="type of model to use")
    parser.add_argument('--bootstrap', type=bool, required=True, help="bootstrap")

    args = parser.parse_args()

    is_train = args.train
    seed = 42
    test_size = 0.2

    print("train" if is_train else "test")

    df = load_data(args.training_set)
    X, y = df.drop("passengers_up", axis=1), df.passengers_up

    preprocess_x, preprocess_y = _preprocess_data(X, y)
    X_train, X_valid, y_train, y_valid = train_test_split(preprocess_x, preprocess_y, test_size=test_size, random_state=seed)

    model, mse, y_pred_on_valid = train_and_evaluate(X_train, X_valid, y_train, y_valid, args.model_type)
    print(f"MSE: {mse}")

    with open(f"model_task1_{args.model_type}.sav", "wb") as f:
        pickle.dump(model, f)
    with open(f"model_task1_{args.model_type}.sav", "rb") as file:
        model = pickle.load(file)

        df = load_data(args.test_set)
        X_test_processed = preprocess_test(df)

        y_pred = model.predict(X_test_processed)

        predictions = pd.DataFrame({'trip_id_unique_station': df['trip_id_unique_station'], 'passenger_up': y_pred})
        predictions.to_csv(args.out, index=False)
