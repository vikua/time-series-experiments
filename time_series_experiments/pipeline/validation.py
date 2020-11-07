from typing import Dict, List
from datetime import timedelta, datetime

import attr
import numpy as np
import pandas as pd

from .dataset import DatasetConfig
from ..utils.data import sliding_window


def compute_backtests(
    start: datetime, end: datetime, k: int, validation_size: float
) -> List[Dict[str, datetime]]:
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


@attr.s
class Backtest(object):
    backtest_number: int = attr.ib()
    backtest_config: Dict[str, datetime] = attr.ib()
    x_train_index: np.ndarray = attr.ib()
    y_train_index: np.ndarray = attr.ib()
    x_test_index: np.ndarray = attr.ib()
    y_test_index: np.ndarray = attr.ib()


class BacktestingCrossVal(object):
    def __init__(
        self,
        data: pd.DataFrame,
        config: DatasetConfig,
        forecast_horizon: int,
        feature_derivation_window: int,
        k: int,
        validation_size: float,
        shuffle_train: bool = True,
    ):
        self._data = data
        self._config = config
        self._forecast_horizon = forecast_horizon
        self._feature_derivation_window = feature_derivation_window
        self._k = k
        self._validation_size = validation_size

        if self._k <= 0:
            raise ValueError("k cannot be lte 0")

        self._backtests: List[Dict[str, datetime]] = None
        self._sliding_index: np.ndarray = None

    @property
    def backtests(self) -> List[Dict[str, datetime]]:
        if self._backtests is not None:
            return self._backtests

        start = self._data[self._config.date_col].min()
        end = self._data[self._config.date_col].max()
        self._backtests = compute_backtests(start, end, self._k, self._validation_size)
        return self._backtests

    @property
    def sliding_index(self) -> np.ndarray:
        if self._sliding_index is not None:
            return self._sliding_index

        self._sliding_index = sliding_window(
            np.arange(self._data.shape[0]),
            self._forecast_horizon + self._feature_derivation_window,
        )
        return self._sliding_index

    def __getitem__(self, i) -> Backtest:
        if i >= self._k:
            raise IndexError(
                "Can't get backtest {} when max value is {}".format(i, self._k)
            )

        bt = self.backtests[i]
        dates = self._data[self._config.date_col]
        x_idx = self.sliding_index[:, : self._feature_derivation_window]
        y_idx = self.sliding_index[:, self._feature_derivation_window :]

        start_dates = dates[x_idx[:, 0]]
        end_dates = dates[y_idx[:, -1]]

        train_mask = (start_dates >= bt["train_start"]).values & (
            end_dates <= bt["validation_start"]
        ).values
        test_mask = (start_dates >= bt["validation_start"]).values & (
            end_dates <= bt["validation_end"]
        ).values

        x_train = x_idx[train_mask]
        y_train = y_idx[train_mask]
        x_test = x_idx[test_mask]
        y_test = y_idx[test_mask]

        return Backtest(i, bt, x_train, y_train, x_test, y_test)

    def __iter__(self):
        for i in range(self._k):
            yield self[i]
