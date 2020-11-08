import tensorflow as tf
from tensorflow.keras import backend as K

from ._base import Block


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
