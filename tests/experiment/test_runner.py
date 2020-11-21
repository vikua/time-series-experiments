import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from time_series_experiments.pipeline import Pipeline, ColumnsProcessor, Step
from time_series_experiments.pipeline.dataset import DatasetConfig, VarType
from time_series_experiments.pipeline.validation import BacktestingCrossVal
from time_series_experiments.pipeline.tasks import (
    DateFeatures,
    Wrap,
    OrdCat,
    TargetLag,
)
from time_series_experiments.experiment.runner import TrainingParams, ExperimentRunner
from ..conftest import generate_target


@pytest.fixture
def dataset():
    nrows = 5000
    start = "2018-01-01 00:00:00"
    freq = "1H"
    dates = pd.date_range(start=start, freq=freq, periods=nrows)
    y = generate_target(nrows, freq=freq, start=start)
    return pd.DataFrame({"target": y, "date": dates})


def create_model(num_out):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(num_out, activation="linear"))
    model.compile(loss="mae", optimizer="sgd")
    return model


def test_single_series_runner(dataset):
    config = DatasetConfig(
        "", date_col="date", target_col="target", feature_types={"date": VarType.DATE}
    )
    num_backtests = 3
    cross_val = BacktestingCrossVal(
        dataset, config, k=num_backtests, validation_size=0.2
    )

    date_pipeline = Pipeline(
        steps=[
            Step("derive", DateFeatures()),
            Step(
                "encode",
                ColumnsProcessor(
                    branches=[
                        Step("num", Wrap(StandardScaler()), types=[VarType.NUM]),
                        Step("cat", OrdCat(), types=[VarType.CAT]),
                    ]
                ),
            ),
        ]
    )
    processor = ColumnsProcessor(
        branches=[Step("date_branch", date_pipeline, types=[VarType.DATE])]
    )
    pipeline = Pipeline(
        steps=[
            Step("extract_date_derived", processor),
            Step("extract_lags", TargetLag(order=1, handle_nan="drop")),
        ]
    )

    fw = 12
    epochs = 5
    model = create_model(fw)
    params = TrainingParams(fdw=24, fw=fw, epochs=epochs, batch_size=64)

    runner = ExperimentRunner(
        df=dataset,
        config=config,
        cross_val=cross_val,
        preprocessing_pipeline=pipeline,
        model=model,
        training_params=params,
    )
    result = runner.run_backtesting()

    assert len(result) == num_backtests
    for bt_num in result:
        hist = result[bt_num]
        assert "loss" in hist.history
        assert "val_loss" in hist.history
        assert len(hist.history["loss"]) == epochs
        assert len(hist.history["val_loss"]) == epochs
        assert np.all(np.isfinite(hist.history["loss"]))
        assert np.all(np.isfinite(hist.history["val_loss"]))
