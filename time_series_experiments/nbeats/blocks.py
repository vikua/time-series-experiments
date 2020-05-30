from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class Block(keras.layers.Layer):
    def __init__(
        self,
        units,
        theta_units,
        layers=4,
        stack_id=0,
        activation="relu",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(Block, self).__init__(**kwargs)

        self.units = units
        self.theta_units = theta_units
        self.layers = layers
        self.stack_id = stack_id

        self.activation = keras.activations.get(activation)

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        self.weigts = {}
        self.biases = {}
        self.theta_b_W = None
        self.theta_f_W = None

    def build(self, input_shape):
        super(Block, self).build(input_shape)

        input_dim = input_shape[-1]

        for i in range(self.layers):
            W = self.add_weight(
                name="W_stack_{}_layer_{}".format(self.stack_id, i),
                shape=(input_dim, self.units),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
            )
            b = self.add_weight(
                name="b_stack_{}_layer_{}".format(self.stack_id, i),
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
            self.weigts[i] = W
            self.biases[i] = b

            input_dim = self.units

        self.theta_b_W = self.add_weight(
            name="stack_{}_theta_b_W".format(self.stack_id),
            shape=(self.units, self.theta_units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        self.theta_f_W = self.add_weight(
            name="stack_{}_theta_f_W".format(self.stack_id),
            shape=(self.units, self.theta_units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

    def call(self, inputs):
        outputs = inputs

        for i in range(self.layers):
            outputs = K.dot(outputs, self.weigts[i])
            outputs = K.bias_add(outputs, self.biases[i], data_format="channels_last")
            outputs = self.activation(outputs)

        theta_b_output = K.dot(outputs, self.theta_b_W)
        theta_f_output = K.dot(outputs, self.theta_f_W)

        return theta_b_output, theta_f_output

    def get_config(self):
        config = super(Block, self).get_config()
        config.update(
            {
                "stack_id": self.stack_id,
                "units": self.units,
                "layers": self.layers,
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
                "kernel_constraint": keras.constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": keras.constraints.serialize(self.bias_constraint),
            }
        )
        return config


class GenericBlock(Block):
    def __init__(self, fdw, fw, **kwargs):
        super(GenericBlock, self).__init__(**kwargs)

        self.fdw = fdw
        self.fw = fw

    def build(self, input_shape):
        super(GenericBlock, self).build(input_shape)

        self.backcast_W = self.add_weight(
            name="stack_{}_backcast_W".format(self.stack_id),
            shape=(self.units, self.fdw),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        self.backcast_b = self.add_weight(
            name="stack_{}_backcast_b".format(self.stack_id),
            shape=(self.fdw,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        self.forecast_W = self.add_weight(
            name="stack_{}_forecast_W".format(self.stack_id),
            shape=(self.units, self.fw),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )
        self.forecast_b = self.add_weight(
            name="stack_{}_forecast_b".format(self.stack_id),
            shape=(self.fw,),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
        )

    def call(self, inputs):
        theta_b_output, theta_f_output = super(GenericBlock, self).call(inputs)

        backcast = K.dot(theta_b_output, self.backcast_W)
        backcast = K.bias_add(backcast, self.backcast_b, data_format="channels_last")

        forecast = K.dot(theta_f_output, self.forecast_W)
        forecast = K.bias_add(forecast, self.forecast_b, data_format="channels_last")

        return backcast, forecast

    def get_config(self):
        config = super(GenericBlock, self).get_config()
        config.update({"fdw": self.fdw, "fw": self.fw})
        return config


class TrendBlock(Block):
    def __init__(self, fdw, fw, **kwargs):
        super(TrendBlock, self).__init__(**kwargs)

        self.fdw = fdw
        self.fw = fw

    def call(self, inputs):
        theta_b_output, theta_f_output = super(TrendBlock, self).call(inputs)

        t = K.cast(K.arange(-self.fdw, self.fw, 1) / self.fdw, tf.float32)

        t = K.transpose(K.stack([t ** i for i in range(self.theta_units)], axis=0))

        t_b = t[: self.fdw]
        t_f = t[self.fdw :]

        backcast = K.dot(theta_b_output, K.transpose(t_b))
        forecast = K.dot(theta_f_output, K.transpose(t_f))

        return backcast, forecast

    def get_config(self):
        config = super(TrendBlock, self).get_config()
        config.update({"fdw": self.fdw, "fw": self.fw})
        return config


class SeasonalBlock(Block):
    def __init__(self, fdw, fw, **kwargs):
        super(SeasonalBlock, self).__init__(**kwargs)

        self.fdw = fdw
        self.fw = fw

    def call(self, inputs):
        theta_b_output, theta_f_output = super(SeasonalBlock, self).call(inputs)

        t = K.cast(K.arange(-self.fdw, self.fw, 1) / self.fdw, tf.float32)

        cos_num = self.theta_units // 2
        sin_num = (
            self.theta_units // 2
            if self.theta_units % 2 == 0
            else self.theta_units // 2 + 1
        )

        cos = K.stack([K.cos(2 * np.pi * i * t) for i in range(cos_num)], axis=0)
        sin = K.stack([K.sin(2 * np.pi * i * t) for i in range(sin_num)], axis=0)

        s = K.concatenate([cos, sin], axis=0)
        s_b = s[:, : self.fdw]
        s_f = s[:, self.fdw :]

        backcast = K.dot(theta_b_output, s_b)
        forecast = K.dot(theta_f_output, s_f)

        return backcast, forecast

    def get_config(self):
        config = super(SeasonalBlock, self).get_config()
        config.update({"fdw": self.fdw, "fw": self.fw})
        return config


class BlockTypes(Enum):
    GENERIC = 1
    TREND = 2
    SEASONAL = 3


BLOCKS = {
    BlockTypes.GENERIC: GenericBlock,
    BlockTypes.TREND: TrendBlock,
    BlockTypes.SEASONAL: SeasonalBlock,
}
