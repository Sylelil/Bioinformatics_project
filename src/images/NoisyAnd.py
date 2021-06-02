from tensorflow.keras.layers import Layer
import tensorflow as tf

class NoisyAnd(Layer):
    """Custom NoisyAND layer from the Deep MIL paper"""

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NoisyAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = 10  # fixed, controls the slope of the activation
        print(input_shape)
        self.b = self.add_weight(name='b',
                                 shape=(1, input_shape[3]),
                                 initializer='uniform',
                                 trainable=True)
        super(NoisyAnd, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[1, 2])
        res = (tf.nn.sigmoid(self.a * (mean - self.b)) - tf.nn.sigmoid(-self.a * self.b)) / (
                tf.nn.sigmoid(self.a * (1 - self.b)) - tf.nn.sigmoid(-self.a * self.b))
        return res

    def get_config(self):
        cfg = super().get_config()
        return cfg

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[3]

    def from_config(self):
        return NoisyAnd(2)