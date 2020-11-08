import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from ._base import Block


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
