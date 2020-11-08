from ._base import Stack


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
