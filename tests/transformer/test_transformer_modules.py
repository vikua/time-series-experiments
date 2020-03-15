import random

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from time_series_experiments.transformer.modules import (
    PositionWiseFeedForwardNetwork,
    TransformerEncoder,
    TransformerDecoder,
)
from time_series_experiments.transformer.layers import PositionalEncoding

from time_series_experiments.utils import get_initializer
from time_series_experiments.utils import rmse
from time_series_experiments.utils import create_decoder_inputs

from ..conftest import simple_seq_data, RANDOM_SEED


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


def test_position_wise_ffnn():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs = keras.layers.Flatten()(inputs)
    outputs = PositionWiseFeedForwardNetwork(
        d_model=128,
        dff=512,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(outputs)
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


def test_transformer_encoder():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs, encoder_self_attention = TransformerEncoder(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(inputs)
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
    assert error < 2.0


def test_transformer_decoder():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )
    decoder_inputs_train = create_decoder_inputs(y_train)

    inputs = keras.Input(shape=(fdw, 1))
    targets = keras.Input(shape=(fw, 1))

    outputs, encoder_self_attention = TransformerEncoder(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(inputs)

    outputs, decoder_self_attention, encoder_decoder_attention = TransformerDecoder(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(targets, outputs)

    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(
        fw,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        activation="linear",
    )(outputs)
    model = keras.Model(inputs=[inputs, targets], outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.MeanSquaredError()
    )

    model.fit(
        [x_train, decoder_inputs_train], y_train, epochs=5, batch_size=32, shuffle=False
    )

    decoder_inputs_test = create_decoder_inputs(y_test)
    y_pred = model.predict([x_test, decoder_inputs_test])
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_pred, y_test)
    assert error < 2.0
