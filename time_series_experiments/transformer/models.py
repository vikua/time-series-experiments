import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .modules import TransformerEncoder, TransformerDecoder
from .layers import PaddingLookAheadMask


class Transformer(keras.Model):
    def __init__(
        self,
        num_layers,
        attention_dim,
        num_heads,
        dff=None,
        hidden_activation="linear",
        hidden_kernel_initializer="glorot_uniform",
        attention_kernel_initializer="glorot_uniform",
        pwffn_kernel_initializer="glorot_uniform",
        output_kernel_initializer="glorot_uniform",
        layer_norm_epsilon=0.001,
        dropout_rate=0.0,
    ):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            attention_dim=attention_dim,
            num_heads=num_heads,
            hidden_activation=hidden_activation,
            dff=dff,
            hidden_kernel_initializer=hidden_kernel_initializer,
            attention_kernel_initializer=attention_kernel_initializer,
            pwffn_kernel_initializer=pwffn_kernel_initializer,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout_rate=dropout_rate,
        )
        self.lookahead_mask_layer = PaddingLookAheadMask()
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            attention_dim=attention_dim,
            num_heads=num_heads,
            hidden_activation=hidden_activation,
            attention_kernel_initializer=attention_kernel_initializer,
            pwffn_kernel_initializer=pwffn_kernel_initializer,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout_rate=dropout_rate,
        )
        self.final_layer = keras.layers.Dense(
            1, kernel_initializer=output_kernel_initializer, activation="linear"
        )

        self.fw = None

    def __call__(self, inputs):
        encoder_inputs, decoder_inputs = inputs
        encoder_outputs, encoder_attention = self.encoder(encoder_inputs)
        lookahead_mask = self.lookahead_mask_layer(decoder_inputs)
        outputs, _, _ = self.decoder(
            decoder_inputs, encoder_outputs, lookahead_mask=lookahead_mask
        )
        return self.final_layer(outputs)

    def train_step(self, inputs):
        x, y = inputs

        self.fw = K.int_shape(y)[1]

        with tf.GradientTape() as tape:
            outputs = self(x)
            loss = self.compiled_loss(y, outputs, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        # grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.compiled_metrics.update_state(y, outputs)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        x, y = inputs
        outputs = self(x)
        self.compiled_loss(y, outputs, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, outputs)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, inputs):
        x, decoder_inputs = inputs[0]
        encoder_outputs, encoder_attention = self.encoder(x)

        all_outputs = []
        for step in range(self.fw):
            lookahead_mask = self.lookahead_mask_layer(decoder_inputs)
            outputs, decoder_attention, encoder_decoder_attention = self.decoder(
                decoder_inputs, encoder_outputs, lookahead_mask=lookahead_mask
            )
            outputs = self.final_layer(outputs)
            last_step = outputs[:, -1, :]
            all_outputs.append(last_step)
            decoder_inputs = tf.concat(
                [decoder_inputs, tf.expand_dims(last_step, axis=-1)], axis=1
            )
        pred = tf.concat(all_outputs, axis=1)
        weights = {
            "encoder_attention": encoder_attention,
            "decoder_attention": decoder_attention,
            "encoder_decoder_attention": encoder_decoder_attention,
        }
        return pred, weights
