from tensorflow.keras import backend as K

from ._base import Block


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
