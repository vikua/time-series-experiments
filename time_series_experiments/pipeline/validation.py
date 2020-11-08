from typing import Dict, List
from datetime import timedelta, datetime

import attr
import numpy as np
import pandas as pd

from .dataset import DatasetConfig


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
    train_index: np.ndarray = attr.ib()
    test_index: np.ndarray = attr.ib()


class BacktestingCrossVal(object):
    def __init__(
        self,
        data: pd.DataFrame,
        config: DatasetConfig,
        k: int,
        validation_size: float,
        shuffle_train: bool = False,
    ):
        self._data = data
        self._config = config
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

    def __getitem__(self, i) -> Backtest:
        if i >= self._k:
            raise IndexError(
                "Can't get backtest {} when max value is {}".format(i, self._k)
            )

        bt = self.backtests[i]
        dates = self._data[self._config.date_col]

        idx = np.arange(dates.shape[0])

        train_mask = (dates >= bt["train_start"]).values & (
            dates <= bt["validation_start"]
        ).values
        test_mask = (dates >= bt["validation_start"]).values & (
            dates <= bt["validation_end"]
        ).values

        train_index = idx[train_mask]
        test_index = idx[test_mask]

        return Backtest(i, bt, train_index, test_index)

    def __iter__(self):
        for i in range(self._k):
            yield self[i]
