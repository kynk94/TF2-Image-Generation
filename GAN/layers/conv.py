import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.ops import nn, nn_ops, array_ops


class ConvBase:
    def get_initializer(self, use_weight_scaling, gain, lr_multiplier):
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier
        if use_weight_scaling:
            stddev = 1.0 / lr_multiplier
            return tf.initializers.random_normal(0, stddev)
        return None

    def _check_weight_scaling(self, input_shape):
        if self.use_weight_scaling:
            input_channel = self._get_input_channel(TensorShape(input_shape))
            fan_in = input_channel * np.prod(self.kernel_size)
            self.runtime_coef = self.gain / np.sqrt(fan_in)
            self.runtime_coef *= self.lr_multiplier
            self.kernel = self.kernel * self.runtime_coef

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _update_config(self, config):
        config.update({
            'use_weight_scaling':
                self.use_weight_scaling,
            'gain':
                self.gain,
            'lr_multiplier':
                self.lr_multiplier
        })


class Conv(convolutional.Conv, ConvBase):
    def __init__(self,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            kernel_initializer=self.get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling(input_shape)

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config


class Conv1D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)


class Conv2D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)


class Conv3D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 groups=1,
                 activation=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)


class Conv1DTranspose(convolutional.Conv1DTranspose, ConvBase):
    def __init__(self,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            kernel_initializer=self.get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling(input_shape)

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config


class Conv2DTranspose(convolutional.Conv2DTranspose, ConvBase):
    def __init__(self,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            kernel_initializer=self.get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling(input_shape)

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config


class Conv3DTranspose(convolutional.Conv3DTranspose, ConvBase):
    def __init__(self,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            kernel_initializer=self.get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling(input_shape)

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config
