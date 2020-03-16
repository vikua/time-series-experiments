import tensorflow as tf
from tensorflow import keras


class MultiHeadAttention(keras.layers.Layer):
    def __init__(
        self,
        attention_dim,
        num_heads,
        temperature=None,
        kernel_initializer=None,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)

        # d_model == attention_dim * num_heads
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.kernel_initializer = kernel_initializer

        self.input_dim = None
        self.seq_len = None

        self.W_Q = None
        self.W_K = None
        self.W_V = None
        self.W_O = None

    def build(self, input_shape):
        _, self.seq_len, self.input_dim = input_shape[0]

        self.W_Q = self.add_weight(
            name="w_q",
            shape=(self.input_dim, self.attention_dim * self.num_heads),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        self.W_K = self.add_weight(
            name="w_k",
            shape=(self.input_dim, self.attention_dim * self.num_heads),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        self.W_V = self.add_weight(
            name="w_v",
            shape=(self.input_dim, self.attention_dim * self.num_heads),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        self.W_O = self.add_weight(
            name="w_o",
            shape=(
                self.attention_dim * self.num_heads,
                self.attention_dim * self.num_heads,
            ),
            initializer=self.kernel_initializer,
            trainable=True,
        )

        super(MultiHeadAttention, self).build(input_shape)

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        temperature = self.temperature
        if temperature is None:
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            temperature = tf.math.sqrt(dk)

        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (1 - mask) * -1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, inputs, mask=None):
        # each input shape (b, seq_len, input_dim)
        q, k, v = inputs

        # apply weights
        # q, k, v shape (b, seq_len, attention_dim * num_heads)
        q = tf.tensordot(q, self.W_Q, axes=(-1, 0))
        k = tf.tensordot(k, self.W_K, axes=(-1, 0))
        v = tf.tensordot(v, self.W_V, axes=(-1, 0))

        # split into heads, shape (num_heads, b, seq_len, attention_dim)
        q = tf.stack(tf.split(q, self.num_heads, axis=-1))
        k = tf.stack(tf.split(k, self.num_heads, axis=-1))
        v = tf.stack(tf.split(v, self.num_heads, axis=-1))

        # transpose q, k, v into (b, num_heads, seq_len, attention_dim)
        q = tf.transpose(q, perm=(1, 0, 2, 3))
        k = tf.transpose(k, perm=(1, 0, 2, 3))
        v = tf.transpose(v, perm=(1, 0, 2, 3))

        x, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        x = tf.transpose(x, perm=(0, 2, 1, 3))
        x = tf.reshape(
            x, shape=(tf.shape(x)[0], self.seq_len, self.attention_dim * self.num_heads)
        )
        x = tf.matmul(x, self.W_O)
        return x, attention_weights

    def get_config(self):
        return {
            "attention_dim": self.attention_dim,
            "num_heads": self.num_heads,
            "temperature": self.temperature,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
        }


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.units = units

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        seq_len = shape[1]

        position_dims = tf.range(seq_len)[:, tf.newaxis]
        input_dims = tf.range(self.units)[tf.newaxis, :]

        angle_rates = 1 / tf.pow(
            10000.0, tf.cast((2 * (input_dims // 2)) / self.units, tf.float32)
        )
        angle_rads = tf.cast(position_dims, tf.float32) * angle_rates

        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        pos_encoding = tf.repeat(pos_encoding, batch_size, axis=0)
        return pos_encoding

    def get_config(self):
        return {"units": self.units}


class PaddingLookAheadMask(keras.layers.Layer):
    def call(self, inputs):
        size = tf.shape(inputs)[1]

        look_ahead_mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        # padding mask is not used yets, so just generating ones
        padding_mask = tf.cast(tf.ones(tf.shape(inputs)[0:2]), tf.float32)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]

        return tf.minimum(look_ahead_mask, padding_mask)
