import pytest
import pandas as pd

from time_series_experiments.pipeline.validation import BacktestingCrossVal
from time_series_experiments.pipeline.dataset import DatasetConfig


@pytest.fixture
def data():
    return pd.DataFrame(
        {"date": pd.date_range(start="2020-01-1", freq="1H", periods=5000)}
    )


def test_time_series_cross_validation_backtests(data):
    cross_val = BacktestingCrossVal(
        data=data,
        config=DatasetConfig("", "date", "", "", {}),
        forecast_horizon=24,
        feature_derivation_window=168,
        k=3,
        validation_size=0.3,
    )
    assert cross_val.backtests is not None
    assert len(cross_val.backtests) == 3
    assert cross_val.backtests == [
        {
            "validation_end": pd.Timestamp("2020-07-27 07:00:00", freq="H"),
            "validation_start": pd.Timestamp("2020-07-06 11:06:00", freq="H"),
            "train_start": pd.Timestamp("2020-02-11 15:48:00", freq="H"),
        },
        {
            "validation_end": pd.Timestamp("2020-07-06 11:06:00", freq="H"),
            "validation_start": pd.Timestamp("2020-06-15 15:12:00", freq="H"),
            "train_start": pd.Timestamp("2020-01-21 19:54:00", freq="H"),
        },
        {
            "validation_end": pd.Timestamp("2020-06-15 15:12:00", freq="H"),
            "validation_start": pd.Timestamp("2020-05-25 19:18:00", freq="H"),
            "train_start": pd.Timestamp("2020-01-01 00:00:00", freq="H"),
        },
    ]


@pytest.mark.parametrize("k", [1, 3, 5])
def test_time_series_cross_validion(data, k):
    cross_val = BacktestingCrossVal(
        data=data,
        config=DatasetConfig("", "date", "", "", {}),
        forecast_horizon=24,
        feature_derivation_window=168,
        k=k,
        validation_size=0.3,
    )
    dates = data["date"].values
    for i in range(k):
        backtest = cross_val[i]
        assert backtest.backtest_number == i
        assert backtest.backtest_config == cross_val.backtests[i]

        x_train = backtest.x_train_index
        y_train = backtest.y_train_index
        x_test = backtest.x_test_index
        y_test = backtest.y_test_index

        assert dates[x_train].min() >= backtest.backtest_config["train_start"]
        assert dates[x_train].max() <= backtest.backtest_config["validation_start"]

        assert dates[y_train].min() >= backtest.backtest_config["train_start"]
        assert dates[y_train].max() <= backtest.backtest_config["validation_start"]

        assert dates[x_test].min() >= backtest.backtest_config["validation_start"]
        assert dates[x_test].max() <= backtest.backtest_config["validation_end"]

        assert dates[y_test].min() >= backtest.backtest_config["validation_start"]
        assert dates[y_test].max() <= backtest.backtest_config["validation_end"]


@pytest.mark.parametrize("k", [1, 3, 5])
def test_time_series_cross_val_interable(data, k):
    cross_val = BacktestingCrossVal(
        data=data,
        config=DatasetConfig("", "date", "", "", {}),
        forecast_horizon=24,
        feature_derivation_window=168,
        k=k,
        validation_size=0.3,
    )
    dates = data["date"].values
    for backtest in cross_val:
        x_train = backtest.x_train_index
        y_train = backtest.y_train_index
        x_test = backtest.x_test_index
        y_test = backtest.y_test_index

        assert dates[x_train].min() >= backtest.backtest_config["train_start"]
        assert dates[x_train].max() <= backtest.backtest_config["validation_start"]

        assert dates[y_train].min() >= backtest.backtest_config["train_start"]
        assert dates[y_train].max() <= backtest.backtest_config["validation_start"]

        assert dates[x_test].min() >= backtest.backtest_config["validation_start"]
        assert dates[x_test].max() <= backtest.backtest_config["validation_end"]

        assert dates[y_test].min() >= backtest.backtest_config["validation_start"]
        assert dates[y_test].max() <= backtest.backtest_config["validation_end"]
