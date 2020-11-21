import numpy as np
import tensorflow as tf

from .data import sliding_window, TaskData


class TimeSeriesSequence(tf.keras.utils.Sequence):
    def __init__(self, data: TaskData, fdw: int, fw: int, batch_size: int):
        self._data = data
        self._fdw = fdw
        self._fw = fw
        self._batch_size = batch_size

        self._idx = None

    @property
    def idx(self):
        if self._idx is not None:
            return self._idx
        self._idx = sliding_window(
            np.arange(self._data.X.shape[0]), self._fdw + self._fw
        )
        return self._idx

    def __len__(self):
        return int(np.ceil(self.idx.shape[0] / float(self._batch_size)))

    def __getitem__(self, i):
        start = i * self._batch_size
        end = start + self._batch_size

        batch_idx = self.idx[start:end]
        x_idx = batch_idx[:, : self._fdw]
        y_idx = batch_idx[:, self._fdw :]

        x_batch = self._data.X[x_idx]
        y_batch = self._data.y[y_idx]
        return x_batch, y_batch
