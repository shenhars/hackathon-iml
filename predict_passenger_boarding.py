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

    # feature_evaluation(df, y)

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
    num_clusters = 50
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

    return best_model, mse, y_pred_on_valid, best_model.score(X_valid, y_valid)


def plot_variance_between_y_true_and_y_pred(y_true, y_pred):
    plt.plot(y_true, color='red', label='Real Values')
    plt.plot(y_pred, color='blue', label='pred Values')
    plt.title(f"Real Values vs Predicted Values")
    plt.legend()
    plt.show()


def plot_results(y_true, y_pred, title="Model Predictions vs Actual Values"):
    """
    Plots the comparison between the actual values and predicted values.

    Args:
    - y_true (pd.Series or np.ndarray): True values.
    - y_pred (pd.Series or np.ndarray): Predicted values.
    - title (str): Title for the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Line plot
    axes[0].plot(y_true, color='red', label='Actual Values')
    axes[0].plot(y_pred, color='blue', label='Predicted Values')
    axes[0].set_title('Actual vs Predicted Values (Line Plot)')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    # Scatter plot
    axes[1].scatter(y_true, y_pred, color='purple', s=10)
    axes[1].set_title('Actual vs Predicted Values (Scatter Plot)')
    axes[1].set_xlabel('Actual Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

    # Error distribution (Histogram)
    errors = y_pred - y_true
    axes[2].hist(errors, bins=50, color='gray', edgecolor='black')
    axes[2].set_title('Prediction Errors (Histogram)')
    axes[2].set_xlabel('Prediction Error')
    axes[2].set_ylabel('Frequency')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Example usage:
# plot_results(y_valid, y_pred_on_valid)

def plot_predictions(y_true, y_pred, mse):
    plt.figure()
    plt.scatter(y_true, color='red', label='Real Values')
    plt.scatter(y_pred, color='blue', label='Predicted Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f"Real Values vs Predicted Values with (MSE = {mse})")
    plt.legend()
    plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True, help="path of the output file as required in the task description")
    parser.add_argument('--model_type', type=str, required=False, default='xgb', choices=['base', 'ridge', 'rf', 'gb', 'xgb', 'lgb'], help="type of model to use")

    args = parser.parse_args()

    seed = 42
    test_size = 0.02

    df = load_data(args.training_set)
    X, y = df.drop("passengers_up", axis=1), df.passengers_up

    preprocess_x, preprocess_y = _preprocess_data(X, y)
    X_train, X_valid, y_train, y_valid = train_test_split(preprocess_x, preprocess_y, test_size=test_size, random_state=seed)

    model, mse, y_pred_on_valid, score = train_and_evaluate(X_train, X_valid, y_train, y_valid, args.model_type)
    print(f"MSE: {mse}")
    print(f"SCORE: {score}")

    plot_results(y_valid, y_pred_on_valid)

    with open(f"model_task1_{args.model_type}.sav", "wb") as f:
        pickle.dump(model, f)
    with open(f"model_task1_{args.model_type}.sav", "rb") as file:
        model = pickle.load(file)

        df = load_data(args.test_set)
        X_test_processed = preprocess_test(df)

        y_pred = model.predict(X_test_processed)

        predictions = pd.DataFrame({'trip_id_unique_station': df['trip_id_unique_station'], 'passenger_up': np.round(y_pred)})
        predictions.to_csv(args.out, index=False)
