from tensorflow import keras


class NBeatS(keras.Model):
    def __init__(
        self, num_stacks, stack_definitions,
    ):
        super(NBeatS, self).__init__()

        self.num_stacks = num_stacks
        self.stack_definitions = stack_definitions
