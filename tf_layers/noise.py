"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


class NoiseBase(tf.keras.layers.Layer):
    def __init__(self,
                 strength=0.0,
                 channel_same=True,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.supports_masking = True
        self.init_strength = strength
        self.channel_same = channel_same

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if self.channel_same and len(input_shape) > 3:
            if tf.keras.backend.image_data_format() == 'channels_first':
                input_shape = [input_shape[0], 1, *input_shape[2:]]
            else:
                input_shape = [*input_shape[:-1], 1]
            input_shape = tf.TensorShape(input_shape)
        self.noise_shape = input_shape[1:]
        self.strength = self.add_weight(
            name='strength',
            shape=(),
            initializer=tf.initializers.Constant(self.init_strength),
            trainable=True,
            dtype=self.dtype)
        self._noise_op = getattr(self, '_noise_op', None)

    def call(self, inputs, training=None):
        return tf.keras.backend.in_train_phase(
            self._noise_op(inputs), inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'init_strength': self.init_strength,
            'channel_same': self.channel_same
        })
        return config

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


class GaussianNoise(NoiseBase):
    def __init__(self,
                 stddev=1.0,
                 strength=0.0,
                 channel_same=True,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(strength=strength,
                         channel_same=channel_same,
                         trainable=trainable,
                         name=name,
                         **kwargs)
        self.stddev = stddev

    def build(self, input_shape):
        super().build(input_shape)

        def noise_op(inputs):
            shape = (tf.shape(inputs)[0], *self.noise_shape)
            noise = tf.random.normal(shape=shape,
                                     mean=0.0,
                                     stddev=self.stddev,
                                     dtype=self.dtype)
            return inputs + noise * self.strength
        self._noise_op = noise_op

    def get_config(self):
        config = super().get_config()
        config.update({
            'stddev': self.stddev
        })
        return config


class UniformNoise(NoiseBase):
    def __init__(self,
                 minval=-1.0,
                 maxval=1.0,
                 strength=0.0,
                 channel_same=True,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(strength=strength,
                         channel_same=channel_same,
                         trainable=trainable,
                         name=name,
                         **kwargs)
        self.minval = minval
        self.maxval = maxval

    def build(self, input_shape):
        super().build(input_shape)

        def noise_op(inputs):
            shape = (tf.shape(inputs)[0], *self.noise_shape)
            noise = tf.random.uniform(shape=shape,
                                      minval=self.minval,
                                      maxval=self.maxval,
                                      dtype=self.dtype)
            return inputs + noise * self.strength
        self._noise_op = noise_op

    def get_config(self):
        config = super().get_config()
        config.update({
            'minval': self.minval,
            'maxval': self.maxval
        })
        return config
