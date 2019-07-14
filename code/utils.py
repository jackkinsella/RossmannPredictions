import numpy as np
import csv


def compare_train_test_error(X_train, X_test, y_train, y_test, model):
    """See if the model is overfitting"""
    train_error = root_mean_square_percentage_error(
        model.predict(X_train), y_train)
    test_error = root_mean_square_percentage_error(
        model.predict(X_test), y_test)
    return train_error - test_error


def log_test_results(settings, error, train_test_error_difference):
    with open('test_results.csv', mode='a') as test_results:
        writer = csv.writer(
            test_results, delimiter=',', quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )

        writer.writerow(['Model', 'Data Description',
                         "Model Description", "Error",
                         "Train test error difference"
                         ])
        writer.writerow(
            [settings["Model"], settings["Data Description"],
                str(settings["Model Description"]),
                error, train_test_error_difference
             ]
        )


def root_mean_square_percentage_error(y_pred, y_true):
    non_zero_indexes = np.where(y_true != 0)
    y_true_no_zeros = y_true.values[non_zero_indexes]
    y_pred_no_zeros = y_pred[non_zero_indexes]

    loss = np.sqrt(
        np.mean(
            np.square(
                ((y_true_no_zeros - y_pred_no_zeros) / y_true_no_zeros)
            ), axis=0)
    )

    return loss * 100
