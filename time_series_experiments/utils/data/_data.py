from datetime import timedelta

import numpy as np


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
    def __init__(self, data, fdw, fw, k=2, validation_size=0.2):
        self._data = data
        self._fdw = fdw
        self._fw = fw
        self._k = k
        self._validation_size = validation_size

        self._backtests = None
        self._x_idx = None
        self._y_idx = None

        self._initialize()

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
        train_mask = (start_dates >= bt["train_start"]) & (
            start_dates <= bt["validation_start"]
        )
        test_mask = (start_dates >= bt["validation_start"]) & (
            start_dates <= bt["validation_end"]
        )
        x_train = self._x_idx[train_mask]
        y_train = self._y_idx[train_mask]
        x_test = self._x_idx[test_mask]
        y_test = self._y_idx[test_mask]
        return x_train, y_train, x_test, y_test

    def __iter__(self):
        for i in range(self._k):
            yield self[i]
