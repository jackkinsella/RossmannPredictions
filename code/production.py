import pandas as pd
import joblib
from utils import root_mean_square_percentage_error

import sys

from prepare_data import process_data

persisted_model_filename = 'models/production.pkl'


def make_predictions(csv_path):
    model = joblib.load(persisted_model_filename)
    X = process_data(csv_path)
    y = model.predict(X)
    predictions_filepath = "predictions.csv"
    print(f"Saving predictions to {predictions_filepath}")
    error = root_mean_square_percentage_error(y, X["Sales"])
    print(f"Error is {error}")
    pd.Series(y).to_csv(predictions_filepath, header=False, index=False)


if __name__ == "__main__":
    csv_path = sys.argv[1]
    make_predictions(csv_path)
