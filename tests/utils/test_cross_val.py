import pytest
import numpy as np
import pandas as pd

from time_series_experiments.utils import data


@pytest.fixture
def dates():
    return pd.date_range(start='2020-01-1', freq='1H', periods=5000)


def test_time_series_cross_validation_backtests(dates):
    cross_val = data.TimeSeriesCrossVal(dates, fdw=168, fw=24, k=3, validation_size=0.3)
    assert cross_val.backtests is not None
    assert len(cross_val.backtests) == 3
    assert cross_val.backtests == [
        {
            'validation_end': pd.Timestamp('2020-07-27 07:00:00', freq='H'),
            'validation_start': pd.Timestamp('2020-07-06 11:06:00', freq='H'),
            'train_start': pd.Timestamp('2020-02-11 15:48:00', freq='H')
        },
        {
            'validation_end': pd.Timestamp('2020-07-06 11:06:00', freq='H'),
            'validation_start': pd.Timestamp('2020-06-15 15:12:00', freq='H'),
            'train_start': pd.Timestamp('2020-01-21 19:54:00', freq='H')},
        {
            'validation_end': pd.Timestamp('2020-06-15 15:12:00', freq='H'),
            'validation_start': pd.Timestamp('2020-05-25 19:18:00', freq='H'),
            'train_start': pd.Timestamp('2020-01-01 00:00:00', freq='H')
        }
    ]


@pytest.mark.parametrize('k', [1, 3, 5])
def test_time_series_cross_validion(dates, k):
    cross_val = data.TimeSeriesCrossVal(dates, fdw=168, fw=24, k=k, validation_size=0.3)

    for i in range(k):
        x_train, y_train, x_test, y_test = cross_val[i]

        assert dates[x_train].min() >= cross_val.backtests[i]['train_start']
        assert dates[x_train].max() <= cross_val.backtests[i]['validation_start']

        assert dates[y_train].min() >= cross_val.backtests[i]['train_start']
        assert dates[y_train].max() <= cross_val.backtests[i]['validation_start']

        assert dates[x_test].min() >= cross_val.backtests[i]['validation_start']
        assert dates[x_test].max() <= cross_val.backtests[i]['validation_end']

        assert dates[y_test].min() >= cross_val.backtests[i]['validation_start']
        assert dates[y_test].max() <= cross_val.backtests[i]['validation_end']
