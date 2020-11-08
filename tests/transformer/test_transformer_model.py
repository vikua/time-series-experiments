import random

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from time_series_experiments.transformer import Transformer
from time_series_experiments.utils import (
    get_initializer,
    create_decoder_inputs,
    create_empty_decoder_inputs,
)
from time_series_experiments.utils.metrics import rmse
from ..conftest import simple_seq_data, RANDOM_SEED


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    tf.keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("layer_norm_epsilon", [1e-3, None])
@pytest.mark.parametrize("dff", [None, 32])
def test_transformer(num_layers, layer_norm_epsilon, dff):
    fdw = 28
    fw = 7

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    transformer = Transformer(
        num_layers=num_layers,
        attention_dim=32,
        num_heads=4,
        hidden_activation="linear",
        dff=dff,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        output_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        layer_norm_epsilon=layer_norm_epsilon,
    )

    transformer.compile(
        loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(0.001)
    )

    decoder_inputs = create_decoder_inputs(y_train, go_token=0)
    transformer.fit(
        [x_train, decoder_inputs], y_train, epochs=5, batch_size=32, verbose=1
    )

    dec_inp = create_empty_decoder_inputs(x_test.shape[0], go_token=0)
    y_pred, weights = transformer.predict([x_test, dec_inp])
    assert rmse(y_test, y_pred) < 1.1

    assert len(weights["encoder_attention"]) == num_layers
    assert len(weights["decoder_attention"]) == num_layers
    assert len(weights["encoder_decoder_attention"]) == num_layers
