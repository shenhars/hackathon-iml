import pickle
from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def _preprocess_data(X: pd.DataFrame, is_train: bool = True):
    """
    preprocess the data
    """
    df = X.drop_duplicates()
    df = split_into_areas(df)
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station', 'alternative', 'trip_id_unique_station',
             'trip_id', 'part', 'cluster', 'arrival_is_estimated'], axis=1, inplace=True)  # remove irelevant columns

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = set_categoriel_feature(df)

    # Check if the station ID is valid (is integer)
    df['station_id_valid'] = df['station_id'].apply(lambda x: isinstance(x, int))
    df = df[df['station_id_valid']]
    df = df.drop(['station_id_valid'], axis=1)

    df.dropna()
    agg = aggregate_train(df)
    agg = agg[agg['trip_duration'] >= 0]
    y = agg['trip_duration']
    agg = agg.drop(['trip_duration', 'trip_id_unique', 'station_index', 'passengers_up', 'passengers_continue',
                    'arrival_time', 'door_closing_time', 'station_id'], axis=1)
    return agg, y


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
    # stations_ids = pd.get_dummies(df['station_id'], prefix='station_id_')
    # df = pd.concat([df, stations_ids], axis=1)

    df.drop(['direction', 'line_id'], axis=1, inplace=True)

    boolean_cols = df.select_dtypes(include=['bool']).columns
    df[boolean_cols] = df[boolean_cols].astype(int)
    return df


def preprocess_test(df: pd.DataFrame):
    df = df.drop_duplicates()
    df = split_into_areas(df)
    df.drop(['latitude', 'longitude', 'station_name', 'trip_id_unique_station', 'alternative', 'trip_id_unique_station',
             'trip_id', 'part', 'cluster', 'arrival_is_estimated'], axis=1, inplace=True)  # remove irelevant columns

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    df = set_categoriel_feature(df)

    df.dropna()
    agg = aggregate_test(df)
    agg = agg.drop(['trip_id_unique', 'station_index', 'passengers_up', 'passengers_continue',
                    'arrival_time', 'door_closing_time', 'station_id'], axis=1)
    return agg


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


def split_into_areas(df):
    # Standardize the longitude and latitude
    scaler = StandardScaler()
    df[['longitude_std', 'latitude_std']] = scaler.fit_transform(df[['longitude', 'latitude']])

    # Apply K-Means clustering
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df['cluster_'] = kmeans.fit_predict(df[['longitude_std', 'latitude_std']])

    # Convert cluster labels to one-hot encoding
    df = pd.concat([df, pd.get_dummies(df['cluster_'], prefix='cluster')], axis=1)

    # Drop unnecessary columns
    df = df.drop(['longitude_std', 'latitude_std', 'cluster_'],axis = 1)

    return df


def calculate_trip_duration(group):
    first_station_time = group.loc[group['station_index'] == 1, 'arrival_time'].values[0]
    last_station_time = group['arrival_time'].values[-1]
    return (last_station_time - first_station_time)/ pd.Timedelta(minutes=1)


def aggregate_train(df):
    # Group by trip_id and calculate the trip duration
    df = df.loc[df["arrival_time"].dropna().index]
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], format='%H:%M:%S')
    trip_duration = df.groupby('trip_id_unique').apply(calculate_trip_duration).reset_index(name='trip_duration')

    result1 = df.groupby('trip_id_unique').agg(lambda x: x.iloc[0]).reset_index()
    result = df.groupby('trip_id_unique').agg(
        total_passengers=('passengers_up', 'sum'),
        total_continue_passengers=('passengers_continue', 'sum'),
        number_of_stations=('station_index', 'count')
        # direction1=('direction_1', 'first')
    ).reset_index()

    result = pd.merge(result, trip_duration, on='trip_id_unique')
    result= pd.merge(result, result1, on='trip_id_unique')
    return result


def aggregate_test(df):
    # Group by trip_id and calculate the trip duration
    result1 = df.groupby('trip_id_unique').agg(lambda x: x.iloc[0]).reset_index()
    result = df.groupby('trip_id_unique').agg(
        total_passengers=('passengers_up', 'sum'),
        total_continue_passengers=('passengers_continue', 'sum'),
        number_of_stations=('station_index', 'count')
    ).reset_index()

    result = pd.merge(result, result1, on='trip_id_unique')
    return result


def load_data(path, encoding='ISO-8859-8'):
    return pd.read_csv(path, encoding=encoding)


def preprocess_data(X, is_train=True):
    if is_train:
        return _preprocess_data(X, is_train=True)
    return preprocess_test(X)


def plot_predictions(y_true, y_pred, mse):
    plt.plot(y_true, color='red', label='Real Values')
    plt.plot(y_pred, color='blue', label='Predicted Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f"Real Values vs Predicted Values with (MSE = {mse})")
    plt.legend()
    plt.show()

def plot_variance_between_y_true_and_y_pred(y_true, y_pred):
    plt.plot(y_true, color='red', label='Real Values')
    plt.plot(y_pred, color='blue', label='pred Values')
    plt.title(f"Real Values vs Predicted Values")
    plt.legend()
    plt.show()

def train_and_evaluate(X_train, X_valid, y_train, y_valid, model_type, poly=None):
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

    return best_model, mse, y_pred_on_valid, mse


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
    seed = 0
    test_size = 0.2

    print("train" if is_train else "test")

    df = load_data(args.training_set)

    logging.info("preprocessing train...")
    preprocess_x, preprocess_y = preprocess_data(df)
    X_train, X_valid, y_train, y_valid = train_test_split(preprocess_x, preprocess_y, test_size=test_size,
                                                          random_state=seed)

    X_train, X_valid, y_train, y_valid = \
                            (np.array(X_train), np.array(X_valid), np.array(y_train), np.array(y_valid))

    model, mse, y_pred_on_valid, score = train_and_evaluate(X_train, X_valid, y_train, y_valid, args.model_type)
    print(f"mse={mse}")
    print(f"score={score}")
    plot_variance_between_y_true_and_y_pred(y_valid, y_pred_on_valid)

    # save the model
    with open(f"model_task1.sav", "wb") as f:
        pass
        pickle.dump(model, f)

    with open("model_task1.sav", "rb") as file:
        pass
        model = pickle.load(file)
        df = load_data(args.test_set)
        logging.info("preprocessing test...")
        X_test_processed = preprocess_data(df, is_train=False)

        logging.info("predicting...")
        y_pred = model.predict(X_test_processed)

        logging.info(f"predictions saved to {args.out}")
        predictions = pd.DataFrame(
            {'trip_id_unique': X_test_processed['trip_id_unique'], 'trip_duration_in_minutes': y_pred})
        predictions.to_csv(args.out, index=False)
