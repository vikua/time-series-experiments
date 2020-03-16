from tensorflow import keras

from .layers import MultiHeadAttention


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


class TransformerEncoder(object):
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

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)

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


class TransformerDecoder(object):
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

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)

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
            [outputs, encoder_outputs, encoder_outputs], mask=padding_mask
        )
        mha2_outputs = self.dropout2(mha2_outputs)
        outputs = keras.layers.add([outputs, mha2_outputs])
        outputs = self.layernorm2(outputs)

        ffn_outputs = self.pwffn(outputs)
        ffn_outputs = self.dropout3(ffn_outputs)
        outputs = keras.layers.add([outputs, ffn_outputs])
        outputs = self.layernorm3(outputs)

        return outputs, decoder_self_attention, encoder_decoder_attention
