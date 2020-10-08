from enum import Enum

from tensorflow import keras

from .blocks import BLOCKS


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


class DRESSStack(Stack):
    """ Doubly REsidual Stacking approach from the paper.
    """

    def call(self, inputs):
        backcast = inputs
        forecast = None

        for i in range(len(self.block_types)):
            block = getattr(self, "block_{}".format(i))

            b, f = block(backcast)

            backcast -= b

            if forecast is None:
                forecast = f
            else:
                forecast += f

        return backcast, forecast


class ParallelStack(Stack):
    """ PARALLEL stacking approach from the paper.
    """

    def call(self, inputs):
        backcast = inputs
        forecast = None

        for i in range(len(self.block_types)):
            block = getattr(self, "block_{}".format(i))

            _, f = block(backcast)

            if forecast is None:
                forecast = f
            else:
                forecast += f

        return backcast, forecast


class NoResidualStack(Stack):
    """ NO-RESIDUAL stacking approach from the paper.
    """

    def call(self, inputs):
        backcast = inputs
        forecast = None

        for i in range(len(self.block_types)):
            block = getattr(self, "block_{}".format(i))

            backcast, f = block(backcast)

            if forecast is None:
                forecast = f
            else:
                forecast += f

        return backcast, forecast


class LastForwardStack(Stack):
    """ LAST-FORWARD stacking approach from the paper.
    """

    def call(self, inputs):
        backcast = inputs

        for i in range(len(self.block_types)):
            block = getattr(self, "block_{}".format(i))

            b, forecast = block(backcast)

            backcast -= b

        return backcast, forecast


class NoResidualLastForwardStack(Stack):
    """ NO-RESIDUAL-LAST-FORWARD stacking approach from the paper.
    """

    def call(self, inputs):
        backcast = inputs

        for i in range(len(self.block_types)):
            block = getattr(self, "block_{}".format(i))

            backcast, forecast = block(backcast)

        return backcast, forecast


class ResidualInputStack(Stack):
    """ RESIDUAL-INPUT stacking approach from the paper.
    """

    def call(self, inputs):
        model_input, backcast = inputs
        forecast = None

        for i in range(len(self.block_types)):
            block = getattr(self, "block_{}".format(i))

            b, f = block(backcast)

            backcast = model_input - b

            if forecast is None:
                forecast = f
            else:
                forecast += f

        return backcast, forecast


class StackDef(object):
    def __init__(
        self,
        stack_type,
        block_types,
        block_units,
        block_theta_units,
        block_layers=4,
        block_activation="relu",
        block_kernel_initializer="glorot_uniform",
        block_bias_initializer="zeros",
        block_kernel_regularizer=None,
        block_bias_regularizer=None,
        block_kernel_constraint=None,
        block_bias_constraint=None,
    ):
        self.stack_type = stack_type
        self.block_types = block_types
        self.block_units = block_units
        self.block_theta_units = block_theta_units
        self.block_layers = block_layers
        self.block_activation = block_activation
        self.block_kernel_initializer = block_kernel_initializer
        self.block_bias_initializer = block_bias_initializer
        self.block_kernel_regularizer = block_kernel_regularizer
        self.block_bias_regularizer = block_bias_regularizer
        self.block_kernel_constraint = block_kernel_constraint
        self.block_bias_constraint = block_bias_constraint

    def get_args(self):
        return {
            "block_types": self.block_types,
            "block_units": self.block_units,
            "block_theta_units": self.block_theta_units,
            "block_layers": self.block_layers,
            "block_activation": self.block_activation,
            "block_kernel_initializer": self.block_kernel_initializer,
            "block_bias_initializer": self.block_bias_initializer,
            "block_kernel_regularizer": self.block_kernel_regularizer,
            "block_bias_regularizer": self.block_bias_regularizer,
            "block_kernel_constraint": self.block_kernel_constraint,
            "block_bias_constraint": self.block_bias_constraint,
        }


class StackTypes(Enum):
    NBEATS_DRESS = 0
    PARALLEL = 1
    NO_RESIDUAL = 2
    LAST_FORWARD = 3
    NO_RESIDUAL_LAST_FORWARD = 4
    RESIDUAL_INPUT = 5


STACKS = {
    StackTypes.NBEATS_DRESS: DRESSStack,
    StackTypes.PARALLEL: ParallelStack,
    StackTypes.NO_RESIDUAL: NoResidualStack,
    StackTypes.LAST_FORWARD: LastForwardStack,
    StackTypes.NO_RESIDUAL_LAST_FORWARD: NoResidualLastForwardStack,
    StackTypes.RESIDUAL_INPUT: ResidualInputStack,
}