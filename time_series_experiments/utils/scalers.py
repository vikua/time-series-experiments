import numpy as np


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


class MeanScaler(object):
    def __init__(self, copy=True):
        self.copy = copy
        self.scale = None

    def fit(self, X):
        X = X.astype(np.float)

        self.scale = np.mean(np.abs(X), axis=1, keepdims=True)
        return self

    def _reshaped_scale(self, X_shape):
        scale = self.scale
        dims = len(X_shape) - 1
        new_shape = (scale.shape[0],) + (1,) * dims
        return np.reshape(scale, new_shape)

    def transform(self, X):
        if self.copy:
            X = X.copy()

        X = X.astype(np.float)

        scale = self._reshaped_scale(X.shape)
        X /= scale
        return X

    def inverse_transform(self, X):
        if self.copy:
            X = X.copy()

        X = X.astype(np.float)

        scale = self._reshaped_scale(X.shape)
        X *= scale
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
