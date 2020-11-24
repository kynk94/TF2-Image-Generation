import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape


class Dense(tf.keras.layers.Dense):
    def __init__(self,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier
        if use_weight_scaling:
            stddev = 1.0 / lr_multiplier
            kernel_initializer = tf.initializers.random_normal(0, stddev)
        super().__init__(kernel_initializer=kernel_initializer, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_weight_scaling:
            input_channel = self._get_input_channel(TensorShape(input_shape))
            fan_in = input_channel * np.prod(self.kernel_size)
            self.runtime_coef = self.gain / np.sqrt(fan_in)
            self.runtime_coef *= self.lr_multiplier
            self.kernel = self.kernel * self.runtime_coef

    def get_config(self):
        config = super().get_config()
        config.update({
            'use_weight_scaling':
                self.use_weight_scaling,
            'gain':
                self.gain,
            'lr_multiplier':
                self.lr_multiplier
        })
        return config
