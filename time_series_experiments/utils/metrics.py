import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


def mase(y_test, y_pred, y_baseline_pred):
    mae_baseline = mean_absolute_error(y_test, y_baseline_pred)
    mae_model = mean_absolute_error(y_test, y_pred)
    return mae_model / mae_baseline
