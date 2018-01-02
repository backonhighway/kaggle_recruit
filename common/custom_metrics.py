from sklearn import metrics
import numpy as np


def rmse(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred) ** 0.5


def rmsle(h, y):
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())