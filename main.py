from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import predict_passenger_boarding
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from pprint import pprint

# main.py --training_set ./data/HU.BER/train_bus_schedule.csv --test_set ./data/HU.BER/train_bus_schedule.csv --out ./data/results/output1.csv --train True --model_type base
"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH --train True/False --model_type base/rf/gb

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv --train True --model_type rf

"""


def load_data(path, encoding='ISO-8859-8'):
    return pd.read_csv(path, encoding=encoding)


def preprocess_data(X, y=None, is_train=True):
    if is_train:
        return predict_passenger_boarding._preprocess_data(X, y)
    return predict_passenger_boarding._preprocess_data(X, is_train=False)


def plot_predictions(y_true, y_pred, k, mse):
    plt.plot(y_true, color='red', label='Real Values')
    plt.plot(y_pred, color='blue', label='Predicted Values')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f"Real Values vs Predicted Values with k = {k} (MSE = {mse})")
    plt.legend()
    plt.show()


def train_and_evaluate(X_train, X_valid, y_train, y_valid, model_type, poly=None):
    if model_type == 'base':
        model = LinearRegression()
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


    X_train_processed = X_train
    X_valid_processed = X_valid

    logging.info("training...")
    model.fit(X_train_processed, y_train)
    y_pred_on_valid = model.predict(X_valid_processed)
    mse = mean_squared_error(y_pred_on_valid, y_valid)
    mse = round(mse, 3)

    return model, mse, y_pred_on_valid


def main():
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True, help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True, help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    parser.add_argument('--train', type=bool, required=True, help="is training or test")
    parser.add_argument('--model_type', type=str, required=True, choices=['base', 'rf', 'gb'],
                        help="type of model to use")
    parser.add_argument('--bootstrap', type=bool, required=True, help="bootstrap")

    args = parser.parse_args()

    is_train = args.train
    seed = 42
    test_size = 0.2

    print("train" if is_train else "test")

    if is_train:
        for k in range(2, 3):
            # poly = PolynomialFeatures(degree=k, include_bias=False)

            df = load_data(args.training_set)
            X, y = df.drop("passengers_up", axis=1), df.passengers_up

            preprocess_x, preprocess_y = preprocess_data(X, y)
            logging.info("preprocessing train...")
            X_train, X_valid, y_train, y_valid = train_test_split(preprocess_x, preprocess_y, test_size=test_size,
                                                                  random_state=seed)
            X_train, X_valid, y_train, y_valid = np.array(X_train), np.array(X_valid), np.array(y_train), np.array(
                y_valid)

            model, mse, y_pred_on_valid = train_and_evaluate(X_train, X_valid, y_train, y_valid, args.model_type)
            print(f"k={k}, mse={mse}")
            plot_predictions(y_valid, y_pred_on_valid, k, mse)

    else:
        df = load_data(args.test_set)
        logging.info("preprocessing test...")
        X_test_processed = preprocess_data(df, is_train=False)

        logging.info("predicting...")
        y_pred = model.predict(X_test_processed)

        logging.info(f"predictions saved to {args.out}")
        predictions = pd.DataFrame(
            {'trip_id_unique_station': X_test_processed['trip_id_unique_station'], 'passenger_up': y_pred})
        predictions.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
