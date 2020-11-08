from tensorflow import keras

from ..blocks import BLOCKS


class Stack(keras.layers.Layer):
    def __init__(
        self,
        fdw,
        fw,
        block_types,
        block_units,
        block_theta_units,
        block_layers=4,
        stack_id=0,
        block_activation="relu",
        block_kernel_initializer="glorot_uniform",
        block_bias_initializer="zeros",
        block_kernel_regularizer=None,
        block_bias_regularizer=None,
        block_kernel_constraint=None,
        block_bias_constraint=None,
        **kwargs
    ):
        super(Stack, self).__init__(**kwargs)

        self.fdw = fdw
        self.fw = fw
        self.block_types = block_types
        self.block_units = block_units
        self.block_theta_units = block_theta_units
        self.block_layers = block_layers
        self.stack_id = stack_id

        self.block_activation = block_activation
        self.block_kernel_initializer = block_kernel_initializer
        self.block_bias_initializer = block_bias_initializer
        self.block_kernel_regularizer = block_kernel_regularizer
        self.block_bias_regularizer = block_bias_regularizer
        self.block_kernel_constraint = block_kernel_constraint
        self.block_bias_constraint = block_bias_constraint

        for i, block_type in enumerate(self.block_types):
            block_cls = BLOCKS[block_type]
            block = block_cls(
                fdw=self.fdw,
                fw=self.fw,
                units=self.block_units,
                theta_units=self.block_theta_units,
                layers=self.block_layers,
                stack_id=self.stack_id,
                activation=self.block_activation,
                kernel_initializer=self.block_kernel_initializer,
                kernel_regularizer=self.block_kernel_regularizer,
                kernel_constraint=self.block_kernel_constraint,
                bias_initializer=self.block_bias_initializer,
                bias_regularizer=self.block_bias_regularizer,
                bias_constraint=self.block_bias_constraint,
            )
            setattr(self, "block_{}".format(i), block)

    def get_config(self):
        config = super(Stack, self).get_config()
        config.update(
            {
                "fdw": self.fdw,
                "fw": self.fw,
                "block_types": self.block_types,
                "block_units": self.block_units,
                "block_theta_units": self.block_theta_units,
                "block_layers": self.block_layers,
                "stack_id": self.stack_id,
                "block_activation": self.block_activation,
                "block_kernel_initializer": keras.initializers.serialize(
                    self.block_kernel_initializer
                ),
                "block_bias_initializer": keras.initializers.serialize(
                    self.block_bias_initializer
                ),
                "block_kernel_regularizer": keras.regularizers.serialize(
                    self.block_kernel_regularizer
                ),
                "block_bias_regularizer": keras.regularizers.serialize(
                    self.block_bias_regularizer
                ),
                "block_kernel_constraint": keras.constraints.serialize(
                    self.block_kernel_constraint
                ),
                "block_bias_constraint": keras.constraints.serialize(
                    self.block_bias_constraint
                ),
            }
        )
