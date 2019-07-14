import numpy as np


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
