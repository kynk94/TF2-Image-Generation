import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from .conv import Conv


class ConvAttention(tf.keras.layers.Layer):
    def __init__(self,
                 n_head=8,
                 causal=False,
                 data_format=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.rank = 2
        self.n_head = n_head
        self.causal = causal
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.use_bias = use_bias
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier
        self.kernel_initializer = kernel_initializer
        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        self.supports_masking = True

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[self._channel_axis])
        self.query_conv = Conv(
            rank=self.rank,
            filters=input_channel // self.n_head,
            kernel_size=1,
            strides=1,
            padding=0,
            activation=None,
            noise=False,
            use_bias=self.use_bias,
            use_weight_scaling=self.use_weight_scaling,
            gain=self.gain,
            lr_multiplier=self.lr_multiplier,
            kernel_initializer=self.kernel_initializer)
        self.key_conv = Conv(
            rank=self.rank,
            filters=input_channel // self.n_head,
            kernel_size=1,
            strides=1,
            padding=0,
            activation=None,
            noise=False,
            use_bias=self.use_bias,
            use_weight_scaling=self.use_weight_scaling,
            gain=self.gain,
            lr_multiplier=self.lr_multiplier,
            kernel_initializer=self.kernel_initializer)
        self.value_conv = Conv(
            rank=self.rank,
            filters=input_channel // self.n_head,
            kernel_size=1,
            strides=1,
            padding=0,
            activation=None,
            noise=False,
            use_bias=self.use_bias,
            use_weight_scaling=self.use_weight_scaling,
            gain=self.gain,
            lr_multiplier=self.lr_multiplier,
            kernel_initializer=self.kernel_initializer)
        self.built = True

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        return self.rank + 1

    def _get_spatial_axes(self):
        channel_axis = self._get_channel_axis()
        spatial_axes = list(range(self.rank + 2))
        del spatial_axes[channel_axis]
        del spatial_axes[0]
        return spatial_axes
