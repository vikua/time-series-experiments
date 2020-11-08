import random

import pytest
import numpy as np
import tensorflow as tf

from time_series_experiments.nbeats.blocks import BlockTypes
from time_series_experiments.nbeats.stacks import StackDef, StackTypes
from time_series_experiments.nbeats import (
    NBEATS,
    NBEATSLastForward,
    NBEATSResidual,
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


def nbeats_test_scenarious():
    args = {
        "block_units": 8,
        "block_theta_units": 8,
        "block_layers": 4,
        "block_kernel_initializer": get_initializer("glorot_uniform", RANDOM_SEED),
        "block_bias_initializer": "zeros",
    }
    tests = [
        [
            StackDef(
                StackTypes.NBEATS_DRESS,
                block_types=[BlockTypes.GENERIC, BlockTypes.TREND],
                **args
            ),
        ],
        [
            StackDef(
                StackTypes.PARALLEL,
                block_types=[BlockTypes.TREND, BlockTypes.GENERIC],
                **args
            ),
            StackDef(
                StackTypes.PARALLEL,
                block_types=[BlockTypes.TREND, BlockTypes.SEASONAL],
                **args
            ),
            StackDef(
                StackTypes.PARALLEL,
                block_types=[BlockTypes.SEASONAL, BlockTypes.SEASONAL],
                **args
            ),
        ],
        [
            StackDef(
                StackTypes.NO_RESIDUAL,
                block_types=[BlockTypes.TREND, BlockTypes.SEASONAL],
                **args
            ),
            StackDef(
                StackTypes.NO_RESIDUAL,
                block_types=[BlockTypes.SEASONAL, BlockTypes.TREND],
                **args
            ),
        ],
    ]
    return tests


def nbeast_last_forward_test_scenarious():
    args = {
        "block_units": 8,
        "block_theta_units": 8,
        "block_layers": 4,
        "block_kernel_initializer": get_initializer("glorot_uniform", RANDOM_SEED),
        "block_bias_initializer": "zeros",
    }
    tests = [
        [
            StackDef(
                StackTypes.LAST_FORWARD,
                block_types=[BlockTypes.GENERIC, BlockTypes.TREND],
                **args
            ),
            StackDef(
                StackTypes.LAST_FORWARD,
                block_types=[BlockTypes.SEASONAL, BlockTypes.TREND],
                **args
            ),
        ],
        [
            StackDef(
                StackTypes.NO_RESIDUAL_LAST_FORWARD,
                block_types=[BlockTypes.GENERIC, BlockTypes.GENERIC],
                **args
            ),
            StackDef(
                StackTypes.NO_RESIDUAL_LAST_FORWARD,
                block_types=[BlockTypes.SEASONAL, BlockTypes.SEASONAL],
                **args
            ),
        ],
    ]
    return tests


def nbeast_residual_test_scenarious():
    args = {
        "block_units": 8,
        "block_theta_units": 8,
        "block_layers": 4,
        "block_kernel_initializer": get_initializer("glorot_uniform", RANDOM_SEED),
        "block_bias_initializer": "zeros",
    }
    tests = [
        [
            StackDef(
                StackTypes.RESIDUAL_INPUT,
                block_types=[BlockTypes.GENERIC, BlockTypes.TREND],
                **args
            ),
            StackDef(
                StackTypes.RESIDUAL_INPUT,
                block_types=[BlockTypes.SEASONAL, BlockTypes.TREND],
                **args
            ),
        ],
        [
            StackDef(
                StackTypes.RESIDUAL_INPUT,
                block_types=[BlockTypes.GENERIC, BlockTypes.GENERIC],
                **args
            ),
        ],
    ]
    return tests


@pytest.mark.parametrize("stacks", nbeats_test_scenarious())
def test_nbeats_model(stacks):
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    model = NBEATS(fdw=fdw, fw=fw, stacks=stacks)
    assert model.num_stacks == len(stacks)
    assert len(model.layers) - 1 == len(stacks)  # reshape layer is also there

    model.compile(optimizer="adam", loss="mae")
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=True)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test, y_pred)
    assert error < 0.45


@pytest.mark.parametrize("stacks", nbeast_last_forward_test_scenarious())
def test_nbeats_last_forward_model(stacks):
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    model = NBEATSLastForward(fdw=fdw, fw=fw, stacks=stacks)
    assert model.num_stacks == len(stacks)
    assert len(model.layers) - 1 == len(stacks)  # reshape layer is also there

    model.compile(optimizer="adam", loss="mae")
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=True)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test, y_pred)
    assert error < 0.45


def test_nbeats_residual_parameters_validation():
    args = {
        "block_units": 8,
        "block_theta_units": 8,
        "block_layers": 4,
        "block_kernel_initializer": get_initializer("glorot_uniform", RANDOM_SEED),
        "block_bias_initializer": "zeros",
    }
    with pytest.raises(ValueError) as excinfo:
        NBEATSResidual(
            fdw=28,
            fw=12,
            stacks=[
                StackDef(
                    StackTypes.RESIDUAL_INPUT,
                    block_types=[BlockTypes.GENERIC, BlockTypes.TREND],
                    **args
                ),
                StackDef(
                    StackTypes.LAST_FORWARD,
                    block_types=[BlockTypes.GENERIC, BlockTypes.TREND],
                    **args
                ),
            ],
        )
    assert str(excinfo.value) == (
        "RESIDUAL-INPUT model supports RESIDUAL-INPUT stacks only. "
        "Found: {<StackTypes.LAST_FORWARD: 3>}"
    )


@pytest.mark.parametrize("stacks", nbeast_residual_test_scenarious())
def test_nbeats_residual_input_model(stacks):
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    model = NBEATSResidual(fdw=fdw, fw=fw, stacks=stacks)
    assert model.num_stacks == len(stacks)
    assert len(model.layers) - 1 == len(stacks)  # reshape layer is also there

    model.compile(optimizer="adam", loss="mae")
    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=True)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test, y_pred)
    assert error < 0.9
