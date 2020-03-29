import numpy as np
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))


def mase(y_test, y_pred, y_baseline_pred):
    mae_baseline = mean_absolute_error(y_test, y_baseline_pred)
    mae_model = mean_absolute_error(y_test, y_pred)
    return mae_model / mae_baseline


def get_initializer(name, seed):
    if name in ["zero", "ones"]:
        return keras.initializers.get(name)
    else:
        return keras.initializers.get({"class_name": name, "config": {"seed": seed}})


class NoOpScaler(object):
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class ScalerWrapper(object):
    def __init__(self, scaler, log_transform=False):
        self.scaler = scaler
        self.log_transform = log_transform

    def fit(self, data):
        data = np.reshape(data, (-1, 1))
        if self.log_transform:
            data = np.log(data)
        self.scaler.fit(data)

    def transform(self, data):
        if self.log_transform:
            data = np.log(data)
        original_shape = data.shape
        scaled_data = self.scaler.transform(np.reshape(data, (-1, 1)))
        return np.reshape(scaled_data, original_shape)

    def inverse_transform(self, data):
        result = self.scaler.inverse_transform(data)
        if self.log_transform:
            result = np.exp(result)
        return result
