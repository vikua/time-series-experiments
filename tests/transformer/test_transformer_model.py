import random

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from time_series_experiments.transformer import TimeSeriesTransformer
from time_series_experiments.utils import get_initializer
from time_series_experiments.utils import rmse
from ..conftest import simple_seq_data, RANDOM_SEED


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def test_transformer():
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    transformer = TimeSeriesTransformer(
        num_layers=1,
        attention_dim=32,
        num_heads=4,
        linear_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        output_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        epochs=5,
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.MeanSquaredError(),
    )

    transformer.fit(x_train, y_train, verbose=1)
    y_pred = transformer.predict(x_test)
    assert rmse(y_pred, y_test) < 1.0
