import numpy as np
from tensorflow import keras

from .modules import Transformer
from ..utils import create_decoder_inputs


class TimeSeriesTransformer(object):
    def __init__(
        self,
        num_layers,
        attention_dim,
        num_heads,
        dff=None,
        linear_kernel_initializer="glorot_uniform",
        attention_kernel_initializer="glorot_uniform",
        pwffn_kernel_initializer="glorot_uniform",
        output_kernel_initializer="glorot_uniform",
        layer_norm_epsilon=0.001,
        dropout_rate=0.0,
        batch_size=32,
        epochs=1,
        loss="mse",
        optimizer="sgd",
        go_token=0,
    ):
        self.num_layers = num_layers
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dff = dff
        self.linear_kernel_initializer = linear_kernel_initializer
        self.attention_kernel_initializer = attention_kernel_initializer
        self.pwffn_kernel_initializer = pwffn_kernel_initializer
        self.output_kernel_initializer = output_kernel_initializer
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout_rate = dropout_rate

        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.go_token = go_token

        self.fwd = None
        self.input_dim = None
        self.fw = None
        self.output_dim = None

        self.model = None

    def build_model(self):
        inputs = keras.Input(shape=(self.fwd, self.input_dim), name="inputs")
        targets = keras.Input(shape=(None, self.output_dim), name="targets")

        (
            outputs,
            encoder_attention,
            decoder_attention,
            encoder_decoder_attention,
        ) = Transformer(
            num_layers=self.num_layers,
            attention_dim=self.attention_dim,
            num_heads=self.num_heads,
            dff=self.dff,
            linear_kernel_initializer=self.linear_kernel_initializer,
            attention_kernel_initializer=self.attention_kernel_initializer,
            pwffn_kernel_initializer=self.pwffn_kernel_initializer,
            output_kernel_initializer=self.output_kernel_initializer,
            layer_norm_epsilon=self.layer_norm_epsilon,
            dropout_rate=self.dropout_rate,
        )(
            inputs, targets
        )

        return keras.Model(inputs=[inputs, targets], outputs=outputs,)

    def fit(self, X, y, verbose=0):
        self.fdw, self.input_dim = X.shape[1], X.shape[2]
        self.fw = y.shape[1]
        self.output_dim = 1

        self.model = self.build_model()
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        decoder_inputs = create_decoder_inputs(y, go_token=self.go_token)

        self.model.fit(
            [X, decoder_inputs],
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=verbose,
        )

    def predict(self, X):
        y_pred = np.empty((X.shape[0], self.fw,))
        decoder_inputs = np.full((X.shape[0], 1, 1), self.go_token)

        for step in range(self.fw):
            pred = self.model.predict([X, decoder_inputs])
            last_pred = pred[:, -1, :]
            y_pred[:, step] = last_pred.ravel()
            decoder_inputs = np.concatenate(
                [decoder_inputs, np.expand_dims(last_pred, 1)], axis=1
            )
        return y_pred
