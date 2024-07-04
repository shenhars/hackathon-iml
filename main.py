from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import predict_passenger_boarding
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
from pprint import pprint

# main.py --training_set ./data/HU.BER/train_bus_schedule.csv --test_set ./data/HU.BER/train_bus_schedule.csv --out ./data/results/output1.csv --train True
"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""

# implement here your load,preprocess,train,predict,save functions (or any other design you choose)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    parser.add_argument('--train', type=bool, required=True,
                        help="is training or test")

    args = parser.parse_args()

    is_train = args.train
    seed = 42
    test_size = .2

    print("train" if is_train else "test")
    if is_train:
        for k in range(5, 6):
            model = LinearRegression()
            poly = PolynomialFeatures(degree=k, include_bias=False)
            # 1. load the training set (args.training_set)
            df = pd.read_csv(args.training_set, encoding='ISO-8859-8')
            X, y = df.drop("passengers_up", axis=1), df.passengers_up

            # 2. preprocess the training set
            # TODO: check if need to change rthe order of the preproccess and the split
            preprocess_x, preprocess_y = predict_passenger_boarding._preprocess_data(X, y)

            logging.info("preprocessing train...")
            X_train, X_valid, y_train, y_valid = train_test_split(preprocess_x, preprocess_y, test_size=test_size, random_state=seed)
            X_train, X_valid, y_train, y_valid = np.array(X_train), np.array(X_valid), np.array(y_train), np.array(y_valid)

            X_train_processed = poly.fit_transform(X_train)
            X_valid_processed = poly.fit_transform(X_valid)

            #3. train a model
            logging.info("training...")
            model.fit(X_train_processed, y_train)

            y_pred_on_valid = model.predict(X_valid_processed)
            mse = mean_squared_error(y_pred_on_valid, y_valid)
            # take only the 10 first digits of the mse
            mse = round(mse, 3)
            print(f"k={k}, mse={mse}")

            # write a code to plot a graph that show in blue the predictions and in red the real values
            import matplotlib.pyplot as plt

            # Plot the real values
            plt.plot(y_valid, color='red', label='Real Values')

            # Plot the predictions
            plt.plot(y_pred_on_valid, color='blue', label='Predicted Values')

            # Add labels and title
            plt.xlabel('Sample')
            plt.ylabel('Value')
            plt.title(f"Real Values vs Predicted Values with k = {k} (MSE = {mse})")

            # Add a legend
            plt.legend()

            plt.show()

    else:
        # 4. load the test set (args.test_set)

        df = pd.read_csv(args.test_set, encoding='ISO-8859-8')

        # 5. preprocess the test set
        logging.info("preprocessing test...")

        X_test_processed = predict_passenger_boarding._preprocess_data(df, is_train=False)

        # 6. predict the test set using the trained model
        logging.info("predicting...")

        y_pred = model.predict(X_test_processed)

        # 7. save the predictions to args.out
        logging.info("predictions saved to {}".format(args.out))

        predictions = pd.DataFrame({
            'trip_id_unique_station': X_test_processed['trip_id_unique_station'],
            'passenger_up': y_pred
        })

        predictions.to_csv(args.out, index=False)


