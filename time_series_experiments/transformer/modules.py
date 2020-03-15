from tensorflow import keras


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
