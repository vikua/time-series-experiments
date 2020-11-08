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
