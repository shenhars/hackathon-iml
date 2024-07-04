import pickle
from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import prepreccess_duration
import xgboost as xgb
import lightgbm as lgb

def load_data(path, encoding='ISO-8859-8'):
    return pd.read_csv(path, encoding=encoding)


def preprocess_data(X, is_train=True):
    if is_train:
        return prepreccess_duration._preprocess_data(X, is_train=True)
    return prepreccess_duration.preprocess_test(X, is_train=False)


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

    X_train_processed = X_train
    X_valid_processed = X_valid

    logging.info("training...")
    model.fit(X_train_processed, y_train)
    y_pred_on_valid = model.predict(X_valid_processed)
    mse = mean_squared_error(y_pred_on_valid, y_valid)
    score = model.score(X_valid_processed, y_valid)
    mse = round(mse, 3)

    return model, mse, y_pred_on_valid, score


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

    if is_train:
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
            pickle.dump(model, f)

    else:
        with open("model_task1.sav", "rb") as file:
            model = pickle.load(file)
            df = load_data(args.test_set)
            logging.info("preprocessing test...")
            X_test_processed = preprocess_data(df, is_train=False)

            logging.info("predicting...")
            y_pred = model.predict(X_test_processed)

            logging.info(f"predictions saved to {args.out}")
            predictions = pd.DataFrame(
                {'trip_id_unique_station': X_test_processed['trip_id_unique_station'], 'passenger_up': y_pred})
            predictions.to_csv(args.out, index=False)
