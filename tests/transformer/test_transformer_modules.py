import random

import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras

from time_series_experiments.transformer.modules import (
    PositionWiseFeedForwardNetwork,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)
from time_series_experiments.transformer.layers import PaddingLookAheadMask

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


class AllOnesPaddingMask(keras.layers.Layer):
    def call(self, inputs):
        padding_mask = tf.cast(tf.ones(tf.shape(inputs)[0:2]), tf.float32)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        return padding_mask


class TestTransformer(object):
    def __init__(
        self,
        num_layers,
        attention_dim,
        num_heads,
        dff=None,
        hidden_kernel_initializer="glorot_uniform",
        attention_kernel_initializer="glorot_uniform",
        pwffn_kernel_initializer="glorot_uniform",
        output_kernel_initializer="glorot_uniform",
        layer_norm_epsilon=0.001,
        dropout_rate=0.0,
    ):
        self.lookahead_mask = PaddingLookAheadMask()

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            attention_dim=attention_dim,
            num_heads=num_heads,
            dff=dff,
            hidden_kernel_initializer=hidden_kernel_initializer,
            attention_kernel_initializer=attention_kernel_initializer,
            pwffn_kernel_initializer=pwffn_kernel_initializer,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout_rate=dropout_rate,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            attention_dim=attention_dim,
            num_heads=num_heads,
            dff=dff,
            hidden_kernel_initializer=hidden_kernel_initializer,
            attention_kernel_initializer=attention_kernel_initializer,
            pwffn_kernel_initializer=pwffn_kernel_initializer,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout_rate=dropout_rate,
        )

        self.output_layer = keras.layers.Dense(
            1, kernel_initializer=output_kernel_initializer, activation="linear",
        )

    def __call__(self, inputs, targets):
        lookahead_mask = self.lookahead_mask(targets)

        enc_output, enc_attention = self.encoder(inputs)

        dec_output, dec_attention, enc_dec_attention = self.decoder(
            targets, enc_output, lookahead_mask=lookahead_mask
        )

        outputs = self.output_layer(dec_output)

        return outputs, enc_attention, dec_attention, enc_dec_attention


def test_position_wise_ffnn():
    fdw = 28
    fw = 7

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
    error = rmse(y_test, y_pred)
    assert error < 0.5


@pytest.mark.parametrize("residual_type", ["add", "concat"])
def test_transformer_encoder_layer(residual_type):
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    outputs, encoder_self_attention = TransformerEncoderLayer(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type=residual_type,
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
    error = rmse(y_test, y_pred)
    assert error < 2.0


@pytest.mark.parametrize("residual_type", ["add", "concat"])
def test_transformer_encoder_layer_masking(residual_type):
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))
    padding_mask = AllOnesPaddingMask()(inputs)
    outputs, encoder_self_attention = TransformerEncoderLayer(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type=residual_type,
    )(inputs, padding_mask=padding_mask)
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
    error = rmse(y_test, y_pred)
    assert error < 2.0


@pytest.mark.parametrize("residual_type", ["add", "concat"])
def test_transformer_decoder_layer(residual_type):
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

    outputs, encoder_self_attention = TransformerEncoderLayer(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(inputs)

    (
        outputs,
        decoder_self_attention,
        encoder_decoder_attention,
    ) = TransformerDecoderLayer(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type=residual_type,
    )(
        targets, outputs
    )

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
    error = rmse(y_test, y_pred)
    assert error < 2.0


@pytest.mark.parametrize("residual_type", ["add", "concat"])
def test_transformer_decoder_layer_masking(residual_type):
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

    padding_mask = AllOnesPaddingMask()(inputs)
    lookahead_mask = PaddingLookAheadMask()(targets)

    outputs, encoder_self_attention = TransformerEncoderLayer(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(inputs, padding_mask=padding_mask)

    (
        outputs,
        decoder_self_attention,
        encoder_decoder_attention,
    ) = TransformerDecoderLayer(
        attention_dim=attention_dim,
        num_heads=num_heads,
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type=residual_type,
    )(
        targets, outputs, padding_mask=padding_mask, lookahead_mask=lookahead_mask
    )

    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(
        fw,
        kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        activation="linear",
    )(outputs)
    model = keras.Model(inputs=[inputs, targets], outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.01),
        loss=keras.losses.MeanSquaredError(),
        run_eagerly=True,
    )

    model.fit(
        [x_train, decoder_inputs_train], y_train, epochs=5, batch_size=32, shuffle=False
    )

    decoder_inputs_test = create_decoder_inputs(y_test)
    y_pred = model.predict([x_test, decoder_inputs_test])
    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test, y_pred)
    assert error < 2.0


@pytest.mark.parametrize("residual_type", ["add", "concat"])
def test_transformer_encoder(residual_type):
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))

    outputs, encoder_self_attention = TransformerEncoder(
        num_layers=2,
        attention_dim=attention_dim,
        num_heads=num_heads,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type=residual_type,
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
    error = rmse(y_test, y_pred)
    assert error < 3.0


def test_transformer_encoder_custom_positional_encoding():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )

    inputs = keras.Input(shape=(fdw, 1))

    outputs, encoder_self_attention = TransformerEncoder(
        num_layers=2,
        attention_dim=attention_dim,
        num_heads=num_heads,
        pos_encoding_dim=8,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type="concat",
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
    error = rmse(y_test, y_pred)
    assert error < 3.0


def test_transformer_encoder_custom_positional_encoding_error():
    attention_dim = 32
    num_heads = 4

    with pytest.raises(ValueError) as excinfo:
        outputs, encoder_self_attention = TransformerEncoder(
            num_layers=2,
            attention_dim=attention_dim,
            num_heads=num_heads,
            pos_encoding_dim=8,
            hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
            attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
            pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
            residual_type="add",
        )
    expected_msg = (
        "Cannot use custom positional encoding dimensionality 8 and residual_type add"
    )
    assert expected_msg == str(excinfo.value)


@pytest.mark.parametrize("residual_type", ["add", "concat"])
def test_transformer_decoder(residual_type):
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

    padding_mask = AllOnesPaddingMask()(inputs)
    lookahead_mask = PaddingLookAheadMask()(targets)

    outputs, encoder_self_attention = TransformerEncoder(
        num_layers=2,
        attention_dim=attention_dim,
        num_heads=num_heads,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(inputs, padding_mask=padding_mask)

    outputs, decoder_weights, encoder_decoder_weights = TransformerDecoder(
        num_layers=2,
        attention_dim=attention_dim,
        num_heads=num_heads,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type=residual_type,
    )(targets, outputs, padding_mask=padding_mask, lookahead_mask=lookahead_mask)

    outputs = keras.layers.Dense(
        1,
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
    error = rmse(y_test, np.squeeze(y_pred))
    assert error < 2.0


def test_transformer_decoder_custom_positional_encoding():
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

    padding_mask = AllOnesPaddingMask()(inputs)
    lookahead_mask = PaddingLookAheadMask()(targets)

    outputs, encoder_self_attention = TransformerEncoder(
        num_layers=2,
        attention_dim=attention_dim,
        num_heads=num_heads,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
    )(inputs, padding_mask=padding_mask)

    outputs, decoder_weights, encoder_decoder_weights = TransformerDecoder(
        num_layers=2,
        attention_dim=attention_dim,
        num_heads=num_heads,
        pos_encoding_dim=8,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        residual_type="concat",
    )(targets, outputs, padding_mask=padding_mask, lookahead_mask=lookahead_mask)

    outputs = keras.layers.Dense(
        1,
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
    error = rmse(y_test, np.squeeze(y_pred))
    assert error < 3.0


def test_transformer_decoder_custom_positional_encoding_error():
    attention_dim = 32
    num_heads = 4

    with pytest.raises(ValueError) as excinfo:
        outputs, decoder_weights, encoder_decoder_weights = TransformerDecoder(
            num_layers=2,
            attention_dim=attention_dim,
            num_heads=num_heads,
            pos_encoding_dim=8,
            hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
            attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
            pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
            residual_type="add",
        )
    expected_msg = (
        "Cannot use custom positional encoding dimensionality 8 and residual_type add"
    )
    assert expected_msg == str(excinfo.value)


def test_transformer():
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

    outputs, enc_attention, dec_attention, enc_dec_attention = TestTransformer(
        num_layers=1,
        attention_dim=attention_dim,
        num_heads=num_heads,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        output_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        layer_norm_epsilon=1e-6,
        dropout_rate=0.1,
    )(inputs, targets)

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
    error = rmse(y_test, np.squeeze(y_pred))
    assert error < 2.0


def test_transformer_model_predictions():
    fdw = 28
    fw = 7
    attention_dim = 32
    num_heads = 4

    x_train, y_train, x_test, y_test = simple_seq_data(
        nrows=1000, freq="1H", fdw=fdw, fw=fw, test_size=0.2
    )
    decoder_inputs_train = create_decoder_inputs(y_train)

    inputs = keras.Input(shape=(fdw, 1))
    targets = keras.Input(shape=(None, 1))

    outputs, enc_attention, dec_attention, enc_dec_attention = TestTransformer(
        num_layers=1,
        attention_dim=attention_dim,
        num_heads=num_heads,
        hidden_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        attention_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        pwffn_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        output_kernel_initializer=get_initializer("glorot_uniform", RANDOM_SEED),
        layer_norm_epsilon=1e-6,
        dropout_rate=0.1,
    )(inputs, targets)

    model = keras.Model(inputs=[inputs, targets], outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001), loss=keras.losses.MeanSquaredError(),
    )

    model.fit(
        [x_train, decoder_inputs_train], y_train, epochs=5, batch_size=32, shuffle=False
    )

    x_test_example = x_test[0][np.newaxis, :, :]
    decoder_inputs = np.array([0.0])[np.newaxis, :, np.newaxis]

    y_pred = np.empty((fw,))
    for step in range(fw):
        pred = model.predict([x_test_example, decoder_inputs])
        last_pred = pred[:, -1:, :]
        y_pred[step] = last_pred.ravel()
        decoder_inputs = np.concatenate([decoder_inputs, last_pred], axis=1)

    assert np.all(np.isfinite(y_pred))
    error = rmse(y_test[0], np.squeeze(y_pred))
    assert error < 1.0
