import random

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from time_series_experiments.nbeats.blocks import (
    Block,
    GenericBlock,
    TrendBlock,
    SeasonalBlock,
)
from time_series_experiments.utils import get_initializer
from time_series_experiments.utils.metrics import rmse

from ..conftest import simple_seq_data, RANDOM_SEED


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def test_base_block():
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs = keras.layers.Reshape((fdw,))(inputs)
    _, outputs = Block(
        units=8,
        theta_units=8,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(inputs)
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(fw)(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
    )
    model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=False)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test, y_pred)
    assert error < 0.5


@pytest.mark.parametrize(
    "block_cls, expected_rmse",
    [(GenericBlock, 0.5), (TrendBlock, 0.4), (SeasonalBlock, 0.35)],
)
def test_blocks(block_cls, expected_rmse):
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs = keras.layers.Reshape((fdw,))(inputs)
    _, forecast = block_cls(
        fdw=fdw,
        fw=fw,
        units=8,
        theta_units=8,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(outputs)
    model = keras.Model(inputs=inputs, outputs=forecast)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
    )
    model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=False)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test, y_pred)
    assert error < expected_rmse
