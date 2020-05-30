from tensorflow import keras

from .stacks import STACKS, StackTypes


class BaseModel(keras.Model):
    def __init__(self, fdw, fw, stacks):
        super(BaseModel, self).__init__()

        self.num_stacks = len(stacks)
        self.stacks = stacks

        self.reshape = keras.layers.Reshape((fdw,))

        for i, stack_def in enumerate(self.stacks):
            stack_cls = STACKS[stack_def.stack_type]
            stack = stack_cls(fdw=fdw, fw=fw, **stack_def.get_args())
            setattr(self, "stack_{}".format(i), stack)


class NBEATS(BaseModel):
    def call(self, inputs):
        inputs = self.reshape(inputs)
        backcast = inputs

        global_forecast = None

        for i in range(self.num_stacks):
            stack = getattr(self, "stack_{}".format(i))
            backcast, forecast = stack(backcast)

            if global_forecast is None:
                global_forecast = forecast
            else:
                global_forecast += forecast

        return global_forecast


class NBEATSLastForward(BaseModel):
    def call(self, inputs):
        inputs = self.reshape(inputs)
        backcast = inputs

        for i in range(self.num_stacks):
            stack = getattr(self, "stack_{}".format(i))
            backcast, forecast = stack(backcast)
        return forecast


class NBEATSResidual(BaseModel):
    def __init__(self, fdw, fw, stacks):
        super(NBEATSResidual, self).__init__(fdw, fw, stacks)

        stack_types = {s.stack_type for s in self.stacks}
        diff = stack_types.difference({StackTypes.RESIDUAL_INPUT})
        if diff:
            raise ValueError(
                "RESIDUAL-INPUT model supports RESIDUAL-INPUT stacks only. Found: {}".format(
                    diff
                )
            )

    def call(self, inputs):
        inputs = self.reshape(inputs)
        backcast = inputs

        global_forecast = None

        for i in range(self.num_stacks):
            stack = getattr(self, "stack_{}".format(i))
            backcast, forecast = stack([inputs, backcast])

            if global_forecast is None:
                global_forecast = forecast
            else:
                global_forecast += forecast

        return global_forecast
