import numpy as np


class LatestNaiveBaseline(object):
    def __init__(self, fw):
        self.fw = fw

    def fit(self, X, y, **kwargs):
        pass

    def predict(self, X, **kwargs):
        preds = X[:, np.newaxis, -1]
        return np.squeeze(np.tile(preds, (self.fw, 1)))
