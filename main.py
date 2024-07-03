from argparse import ArgumentParser
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
from pprint import pprint


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
    test_size = .1
    model = LinearRegression()


    if is_train:
        # 1. load the training set (args.training_set)
        df = pd.read_csv(args.training_set, encoding='ISO-8859-8')
        X, y = df.drop("passengers_up", axis=1), df.passengers_up

        # 2. preprocess the training set
        # TODO: check if need to change rthe order of the preproccess and the split

        logging.info("preprocessing train...")
        X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size=test_size, random_state=seed)
        X_train_processed = predict_passenger_boarding._preprocess_data(X_train, is_train=True)
        X_valid_processed = predict_passenger_boarding._preprocess_data(X_valid, is_train=True)

        # 3. train a model
        logging.info("training...")
        model.fit(X_train_processed, y_train)

        y_pred_on_valid = model.predict(X_valid_processed)
        mse = mean_squared_error(y_pred_on_valid, y_valid)


    else:
        # 4. load the test set (args.test_set)

        df = pd.read_csv(args.test_set, encoding='ISO-8859-8')

        # 5. preprocess the test set
        logging.info("preprocessing test...")

        X_test_processed = predict_passenger_boarding._preprocess_data(df, is_train=False)

        # 6. predict the test set using the trained model
        logging.info("predicting...")

        y_pred = model.predict(X_test)
        print("DEBUG MODE: mse over test")
        mse = mean_squared_error(y_test, y_pred)
        print(mse)

        # 7. save the predictions to args.out
        logging.info("predictions saved to {}".format(args.out))

        predictions = pd.DataFrame({
            'trip_id_unique_station': X_test_processed['trip_id_unique_station'],
            'passenger_up': y_pred
        })

        predictions.to_csv(args.out, index=False)


