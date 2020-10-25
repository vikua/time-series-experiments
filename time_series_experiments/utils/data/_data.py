from datetime import timedelta

import numpy as np
import tensorflow as tf


def sliding_window(arr, window_size):
    """ Takes and arr and reshapes it into 2D matrix

    Parameters
    ----------
    arr: np.ndarray
        array to reshape
    window_size: int
        sliding window size

    Returns
    -------
    new_arr: np.ndarray
        2D matrix of shape (arr.shape[0] - window_size + 1, window_size)
    """
    (stride,) = arr.strides
    arr = np.lib.index_tricks.as_strided(
        arr,
        (arr.shape[0] - window_size + 1, window_size),
        strides=[stride, stride],
        writeable=False,
    )
    return arr


def train_test_split_index(
    size, fdw_steps, fw_steps, test_size, random_seed, shuffle_train=True
):
    idx = np.arange(size)
    idx = sliding_window(idx, fdw_steps + fw_steps)

    train_size = idx.shape[0] - int(idx.shape[0] * test_size)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    if shuffle_train:
        random_state = np.random.RandomState(random_seed)
        shuffle_idx = random_state.permutation(train_idx.shape[0])
        train_idx = train_idx[shuffle_idx]

    x_train_idx = train_idx[:, :fdw_steps]
    y_train_idx = train_idx[:, fdw_steps:]
    x_test_idx = test_idx[:, :fdw_steps]
    y_test_idx = test_idx[:, fdw_steps:]

    return x_train_idx, y_train_idx, x_test_idx, y_test_idx


def create_decoder_inputs(y, go_token=0):
    assert len(y.shape) == 2

    go_tokens = np.empty((y.shape[0], 1, 1))
    go_tokens.fill(go_token)

    decoder_inputs = y[:, :-1, np.newaxis]
    decoder_inputs = np.concatenate([go_tokens, decoder_inputs], axis=1)
    return decoder_inputs


def compute_backtests(start, end, k=2, validation_size=0.2):
    total_sec = (end - start).total_seconds()
    total_val_sec = total_sec * validation_size
    train_sec = total_sec * (1 - validation_size)
    val_sec = total_val_sec / k
    val_delta = timedelta(seconds=val_sec)
    train_delta = timedelta(seconds=train_sec)

    backtests = []
    last_date = end
    for backtest in range(k):
        val_start = last_date - val_delta
        bt = {
            "validation_end": last_date,
            "validation_start": val_start,
            "train_start": last_date - (train_delta + val_delta),
        }
        last_date = val_start
        backtests.append(bt)
    return backtests


class TimeSeriesCrossVal(object):
    """ Creates backtesting partitions for time series data.
    Returns indexes of X and y for both train and tests subsets of single backtests.

    Parameters
    ----------
    data : pd.Series
        dates series from original dataset
    fdw : int
        feature derivation window
    fw : int
        forecast window
    k : int
        number of partitions (backtests)
    validation_size : float
        percent of data to use for validaton. This percent is taken from the whole dataset and
        and then split between different backtests, thus if number of backtests is 3 and
        validation_size is 0.2, each backtest gets 0.2 / 3 fraction of whole dataset.
    """
    def __init__(self, data, fdw, fw, k=2, validation_size=0.2):
        if k == 0:
            raise ValueError("k can't be 0")
        self._data = data
        self._fdw = fdw
        self._fw = fw
        self._k = k
        self._validation_size = validation_size

        self._backtests = None
        self._x_idx = None
        self._y_idx = None

        self._initialize()

    @property
    def backtests(self):
        return self._backtests

    def _compute_partitions(self):
        start = self._data.min()
        end = self._data.max()
        self._backtests = compute_backtests(
            start, end, k=self._k, validation_size=self._validation_size
        )

    def _compute_index(self):
        idx = sliding_window(np.arange(self._data.shape[0]), self._fdw + self._fw)
        self._x_idx = idx[:, : self._fdw]
        self._y_idx = idx[:, self._fdw :]

    def _initialize(self):
        if self._backtests is None:
            self._compute_partitions()
        if self._x_idx is None or self._y_idx is None:
            self._compute_index()

    def __getitem__(self, i):
        if i >= self._k:
            raise IndexError(
                "Can't get backtest {} when max value is {}".format(i, self._k)
            )

        bt = self._backtests[i]
        start_dates = self._data[self._x_idx[:, 0]]
        end_dates = self._data[self._y_idx[:, -1]]
        train_mask = (start_dates >= bt["train_start"]) & (
            end_dates <= bt["validation_start"]
        )
        test_mask = (start_dates >= bt["validation_start"]) & (
            end_dates <= bt["validation_end"]
        )
        x_train = self._x_idx[train_mask]
        y_train = self._y_idx[train_mask]
        x_test = self._x_idx[test_mask]
        y_test = self._y_idx[test_mask]
        return x_train, y_train, x_test, y_test

    def __iter__(self):
        for i in range(self._k):
            yield self[i]


class TimeSeriesSequence(tf.keras.utils.Sequence):
    def __init__(self, data, x_idx, y_idx, batch_size=128):
        self._data = data
        self._x_idx = x_idx
        self._y_idx = y_idx
        self._batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self._x_idx.shape[0] / float(self._batch_size)))

    def __getitem__(self, i):
        start = i * self._batch_size
        end = start + self._batch_size

        batch_x = self._x_idx[start:end]
        batch_y = self._y_idx[start:end]

        return self._data[batch_x], self._data[batch_y]
