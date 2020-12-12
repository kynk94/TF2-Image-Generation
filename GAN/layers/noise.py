"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import tensorflow as tf


class GaussianNoise(tf.keras.layers.GaussianNoise):
    def __init__(self,
                 stddev=1.0,
                 strength=0.0,
                 channel_same=True,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(stddev=stddev,
                         trainable=trainable,
                         name=name,
                         **kwargs)
        self.strength_val = strength
        self.channel_same = channel_same

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if self.channel_same and len(input_shape) > 3:
            data_format = tf.keras.backend.image_data_format()
            if data_format == 'channels_first':
                input_shape = [input_shape[0], 1, *input_shape[2:]]
            else:
                input_shape = [*input_shape[:-1], 1]
            input_shape = tf.TensorShape(input_shape)
        self.noise_shape = input_shape[1:]
        self.strength = self.add_weight(
            name='strength',
            shape=(),
            initializer=tf.initializers.Constant(self.strength_val),
            trainable=True,
            dtype=self.dtype)

    def call(self, inputs, training=None):
        def noised():
            shape = (tf.shape(inputs)[0], *self.noise_shape)
            noise = tf.random.normal(shape=shape,
                                     mean=0.0,
                                     stddev=self.stddev,
                                     dtype=self.dtype)
            return inputs + noise * self.strength
        return tf.keras.backend.in_train_phase(
            noised, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            'init_strength': self.strength_val,
            'channel_same': self.channel_same
        })
        return config
