import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import normalize_tuple
from .utils import get_layer_config


class ICNR(tf.keras.initializers.Initializer):
    """ICNR Initializer for abstract N-D convolution."""

    def __init__(self,
                 scale=2,
                 initializer='he_normal'):
        self.scale = scale
        self.initializer = tf.keras.initializers.get(initializer)

    def __call__(self, shape, dtype=tf.float32):
        scale = normalize_tuple(self.scale, len(shape) - 2, 'scale')
        if set(scale) == {1}:
            return self.initializer(shape, dtype=dtype)

        # shape: (kernel_size, kernel_size, channels//groups, filters) in Conv2D
        new_shape = tuple(shape[:-1]) + (shape[-1] // np.prod(scale), )
        multiples = (1, ) * (len(shape) - 1) + (np.prod(scale), )
        outputs = self.initializer(new_shape, dtype=dtype)
        return tf.tile(outputs, multiples)

    def get_config(self):
        config = {
            'name': self.__class__.__name__,
            'scale': self.scale,
            'initializer': get_layer_config(self.initializer)
        }
        return config
