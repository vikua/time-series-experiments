from tensorflow import keras

from .layers import (
    MultiHeadAttention,
    PositionalEncoding,
)


class PositionWiseFeedForwardNetwork(object):
    def __init__(
        self,
        d_model,
        dff,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        activation=keras.activations.relu,
    ):
        self.dense_1 = keras.layers.Dense(
            dff,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation=activation,
        )
        self.dense_2 = keras.layers.Dense(
            d_model,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation=None,
        )

    def __call__(self, inputs):
        outputs = self.dense_1(inputs)
        outputs = self.dense_2(outputs)
        return outputs


class TransformerEncoderLayer(object):
    def __init__(
        self,
        attention_dim,
        num_heads,
        dff=None,
        attention_kernel_initializer="glorot_uniform",
        pwffn_kernel_initializer="glorot_uniform",
        layer_norm_epsilon=0.001,
        dropout_rate=0.0,
    ):
        self.mha = MultiHeadAttention(
            attention_dim=attention_dim,
            num_heads=num_heads,
            kernel_initializer=attention_kernel_initializer,
        )

        self.pwffn = PositionWiseFeedForwardNetwork(
            d_model=attention_dim * num_heads,
            dff=dff or attention_dim * num_heads * 4,
            kernel_initializer=pwffn_kernel_initializer,
        )

        if layer_norm_epsilon is not None:
            self.layernorm1 = keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon
            )
            self.layernorm2 = keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon
            )
        else:
            self.layernorm1 = keras.layers.Lambda(lambda x: x)
            self.layernorm2 = keras.layers.Lambda(lambda x: x)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def __call__(self, inputs, padding_mask=None):
        outputs, encoder_self_attention = self.mha(
            [inputs, inputs, inputs], mask=padding_mask
        )
        outputs = self.dropout1(outputs)
        outputs = keras.layers.add([inputs, outputs])
        outputs = self.layernorm1(outputs)

        ffn_outputs = self.pwffn(outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        outputs = keras.layers.add([outputs, ffn_outputs])
        outputs = self.layernorm2(outputs)

        return outputs, encoder_self_attention


class TransformerDecoderLayer(object):
    def __init__(
        self,
        attention_dim,
        num_heads,
        dff=None,
        attention_kernel_initializer="glorot_uniform",
        pwffn_kernel_initializer="glorot_uniform",
        layer_norm_epsilon=0.001,
        dropout_rate=0.0,
    ):
        self.mha1 = MultiHeadAttention(
            attention_dim=attention_dim,
            num_heads=num_heads,
            kernel_initializer=attention_kernel_initializer,
        )
        self.mha2 = MultiHeadAttention(
            attention_dim=attention_dim,
            num_heads=num_heads,
            kernel_initializer=attention_kernel_initializer,
        )

        self.pwffn = PositionWiseFeedForwardNetwork(
            d_model=attention_dim * num_heads,
            dff=dff or attention_dim * num_heads * 4,
            kernel_initializer=pwffn_kernel_initializer,
        )

        if layer_norm_epsilon is not None:
            self.layernorm1 = keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon
            )
            self.layernorm2 = keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon
            )
            self.layernorm3 = keras.layers.LayerNormalization(
                epsilon=layer_norm_epsilon
            )
        else:
            self.layernorm1 = keras.layers.Lambda(lambda x: x)
            self.layernorm2 = keras.layers.Lambda(lambda x: x)
            self.layernorm3 = keras.layers.Lambda(lambda x: x)

        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dropout3 = keras.layers.Dropout(dropout_rate)

    def __call__(self, inputs, encoder_outputs, padding_mask=None, lookahead_mask=None):
        outputs, decoder_self_attention = self.mha1(
            [inputs, inputs, inputs], mask=lookahead_mask
        )
        outputs = self.dropout1(outputs)
        outputs = keras.layers.add([inputs, outputs])
        outputs = self.layernorm1(outputs)

        mha2_outputs, encoder_decoder_attention = self.mha2(
            [outputs, encoder_outputs, encoder_outputs],
            mask=padding_mask
            # [encoder_outputs, encoder_outputs, outputs], mask=padding_mask
        )
        mha2_outputs = self.dropout2(mha2_outputs)
        outputs = keras.layers.add([outputs, mha2_outputs])
        outputs = self.layernorm2(outputs)

        ffn_outputs = self.pwffn(outputs)
        ffn_outputs = self.dropout3(ffn_outputs)
        outputs = keras.layers.add([outputs, ffn_outputs])
        outputs = self.layernorm3(outputs)

        return outputs, decoder_self_attention, encoder_decoder_attention


class TransformerEncoder(object):
    def __init__(
        self,
        num_layers,
        attention_dim,
        num_heads,
        hidden_activation="linear",
        dff=None,
        hidden_kernel_initializer="glorot_uniform",
        attention_kernel_initializer="glorot_uniform",
        pwffn_kernel_initializer="glorot_uniform",
        layer_norm_epsilon=0.001,
        dropout_rate=0.0,
    ):
        self.num_layers = num_layers

        self.hidden = keras.layers.Dense(
            attention_dim * num_heads,
            kernel_initializer=hidden_kernel_initializer,
            activation=hidden_activation,
        )
        self.pos_encoding = PositionalEncoding(attention_dim * num_heads)

        self.encoder_layers = [
            TransformerEncoderLayer(
                attention_dim=attention_dim,
                num_heads=num_heads,
                dff=dff,
                attention_kernel_initializer=attention_kernel_initializer,
                pwffn_kernel_initializer=pwffn_kernel_initializer,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout_rate=dropout_rate,
            )
            for i in range(self.num_layers)
        ]

        self.dropout = keras.layers.Dropout(dropout_rate)

    def __call__(self, inputs, padding_mask=None):
        outputs = self.hidden(inputs)
        pos_enc = self.pos_encoding(outputs)
        outputs = keras.layers.add([outputs, pos_enc])

        outputs = self.dropout(outputs)

        attention_weighs = {}

        for i in range(self.num_layers):
            outputs, weights = self.encoder_layers[i](
                outputs, padding_mask=padding_mask
            )
            attention_weighs["enc_layer_{}".format(i)] = weights

        return outputs, attention_weighs


class TransformerDecoder(object):
    def __init__(
        self,
        num_layers,
        attention_dim,
        num_heads,
        hidden_activation="linear",
        dff=None,
        hidden_kernel_initializer="glorot_uniform",
        attention_kernel_initializer="glorot_uniform",
        pwffn_kernel_initializer="glorot_uniform",
        layer_norm_epsilon=0.001,
        dropout_rate=0.0,
    ):
        self.num_layers = num_layers

        self.hidden = keras.layers.Dense(
            attention_dim * num_heads,
            kernel_initializer=hidden_kernel_initializer,
            activation=hidden_activation,
        )
        self.pos_encoding = PositionalEncoding(attention_dim * num_heads)

        self.decoder_layers = [
            TransformerDecoderLayer(
                attention_dim=attention_dim,
                num_heads=num_heads,
                dff=dff,
                attention_kernel_initializer=attention_kernel_initializer,
                pwffn_kernel_initializer=pwffn_kernel_initializer,
                layer_norm_epsilon=layer_norm_epsilon,
                dropout_rate=dropout_rate,
            )
            for i in range(self.num_layers)
        ]

        self.dropout = keras.layers.Dropout(dropout_rate)

    def __call__(self, inputs, encoder_outputs, padding_mask=None, lookahead_mask=None):
        outputs = self.hidden(inputs)
        pos_enc = self.pos_encoding(outputs)
        outputs = keras.layers.add([outputs, pos_enc])

        outputs = self.dropout(outputs)

        dec_attention_weights = {}
        enc_dec_attention_weights = {}

        for i in range(self.num_layers):
            outputs, dec_weights, enc_dec_weights = self.decoder_layers[i](
                outputs,
                encoder_outputs,
                padding_mask=padding_mask,
                lookahead_mask=lookahead_mask,
            )
            dec_attention_weights["dec_layer_{}".format(i)] = dec_weights
            enc_dec_attention_weights["enc_dec_layer_{}".format(i)] = enc_dec_weights

        return outputs, dec_attention_weights, enc_dec_attention_weights
