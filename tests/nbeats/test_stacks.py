import random

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from time_series_experiments.nbeats.blocks import BlockTypes
from time_series_experiments.nbeats.stacks import (
    DRESSStack,
    ParallelStack,
    NoResidualStack,
    LastForwardStack,
    NoResidualLastForwardStack,
    ResidualInputStack,
)
from time_series_experiments.utils import get_initializer
from time_series_experiments.utils import rmse

from ..conftest import simple_seq_data, RANDOM_SEED


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


@pytest.mark.parametrize(
    "stack_cls, block_types, expected_rmse",
    [
        (DRESSStack, [BlockTypes.GENERIC], 0.5),
        (DRESSStack, [BlockTypes.GENERIC, BlockTypes.TREND], 0.3),
        (DRESSStack, [BlockTypes.GENERIC, BlockTypes.TREND, BlockTypes.SEASONAL], 0.25),
        (ParallelStack, [BlockTypes.TREND], 0.4),
        (ParallelStack, [BlockTypes.SEASONAL, BlockTypes.TREND], 0.5),
        (
            ParallelStack,
            [BlockTypes.SEASONAL, BlockTypes.TREND, BlockTypes.GENERIC],
            0.4,
        ),
        (NoResidualStack, [BlockTypes.GENERIC], 0.5),
        (NoResidualStack, [BlockTypes.GENERIC, BlockTypes.TREND], 0.6),
        (
            NoResidualStack,
            [BlockTypes.GENERIC, BlockTypes.TREND, BlockTypes.SEASONAL],
            0.4,
        ),
        (LastForwardStack, [BlockTypes.GENERIC], 0.45),
        (LastForwardStack, [BlockTypes.GENERIC, BlockTypes.TREND], 0.4),
        (
            LastForwardStack,
            [BlockTypes.GENERIC, BlockTypes.TREND, BlockTypes.SEASONAL],
            0.45,
        ),
        (NoResidualLastForwardStack, [BlockTypes.GENERIC], 0.45),
        (NoResidualLastForwardStack, [BlockTypes.GENERIC, BlockTypes.TREND], 0.3),
        (
            NoResidualLastForwardStack,
            [BlockTypes.GENERIC, BlockTypes.TREND, BlockTypes.SEASONAL],
            0.45,
        ),
        (ResidualInputStack, [BlockTypes.GENERIC], 0.45),
        (ResidualInputStack, [BlockTypes.GENERIC, BlockTypes.TREND], 0.3),
        (
            ResidualInputStack,
            [BlockTypes.GENERIC, BlockTypes.TREND, BlockTypes.SEASONAL],
            0.3,
        ),
    ],
)
def test_stacks(stack_cls, block_types, expected_rmse):
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs = keras.layers.Reshape((fdw,))(inputs)
    stack = stack_cls(
        fdw=fdw,
        fw=fw,
        block_types=block_types,
        block_units=8,
        block_theta_units=8,
        block_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )
    if stack_cls == ResidualInputStack:
        _, forecast = stack([outputs, outputs])
    else:
        _, forecast = stack(outputs)
    model = keras.Model(inputs=inputs, outputs=forecast)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
    )
    model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=False)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test, y_pred)
    assert error < expected_rmse
