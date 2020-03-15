import random

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from time_series_experiments.transformer.layers import (
    MultiHeadAttention,
    PositionalEncoding,
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


def _positional_encoding_reference(seq_len, dims):
    def _angle_vec(pos):
        return [pos / np.power(10000.0, 2 * (u // 2) / dims) for u in range(dims)]

    table = np.array([_angle_vec(pos) for pos in range(seq_len)])
    sines = np.sin(table[:, 0::2])
    cosines = np.cos(table[:, 1::2])

    # instead of concatentation: should be sin on even cos on odd positions
    # table[:, 0::2] = sines
    # table[:, 1::2] = cosines

    table = np.concatenate([sines, cosines], axis=-1)
    return table


def test_multi_head_attention():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs, attention_weights = MultiHeadAttention(
        attention_dim=attention_dim,
        num_heads=num_heads,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )([inputs, inputs, inputs])
    outputs = keras.layers.Reshape((fdw * attention_dim * num_heads,))(outputs)
    outputs = keras.layers.Dense(
        fw,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        activation="linear",
    )(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
    )

    model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=False)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_pred, y_test)
    assert error < 0.5


def test_positional_encoding_table():
    fdw = 28
    fw = 7

    x_train, _, _, _ = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    x_train = tf.convert_to_tensor(x_train[0][np.newaxis, :, :])
    pos = PositionalEncoding(128)
    pos_encoding = pos.call(x_train)
    pos_encoding = tf.squeeze(pos_encoding).numpy()

    reference_encoding = _positional_encoding_reference(fdw, 128)
    assert np.all(np.isclose(pos_encoding, reference_encoding, atol=1e-5))


def test_positional_encoding():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs = PositionalEncoding(8)(inputs)
    outputs = keras.layers.Concatenate()([inputs, outputs])
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(
        fw,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        activation="linear",
    )(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
    )

    model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=False)
    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_pred, y_test)
    assert error < 0.5


def test_positional_encoding_and_attention():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs = PositionalEncoding(8)(inputs)
    outputs = keras.layers.Concatenate()([inputs, outputs])
    outputs, attention_weights = MultiHeadAttention(
        attention_dim=attention_dim,
        num_heads=num_heads,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )([outputs, outputs, outputs])
    outputs = keras.layers.Reshape((fdw * attention_dim * num_heads,))(outputs)
    outputs = keras.layers.Dense(
        fw,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        activation="linear",
    )(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
    )

    model.fit(x_train, y_train, epochs=5, batch_size=32, shuffle=False)

    y_pred = model.predict(x_test)
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_pred, y_test)
    assert error < 0.5
