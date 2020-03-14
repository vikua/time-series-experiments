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

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        temperature = self.temperature
        if temperature is None:
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            temperature = tf.math.sqrt(dk)

        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += mask * -1e9

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
