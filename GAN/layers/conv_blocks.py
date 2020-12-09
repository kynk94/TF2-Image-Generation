"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import normalize_data_format
from tensorflow_addons.layers import SpectralNormalization
from .conv import Conv, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from .conv import UpsampleConv2D, SubPixelConv2D
from .utils import get_activation_layer, get_normalization_layer, get_padding_layer
from .utils import get_layer_config


class BaseBlock(tf.keras.Model):
    """
    Base Block for Convolution Block.

    Base block consists of pad, normalization, and activation layers.
    """

    def __init__(self,
                 rank,
                 data_format=None,
                 padding=0,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_first=False,
                 activation_alpha=0.3,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.data_format = normalize_data_format(data_format)
        self.channel_axis = self._get_channel_axis()
        if normalization_first and activation_first:
            raise ValueError('Only one of `normalization_first` '
                             'or `activation_first` can be True.')
        self.normalization_first = normalization_first
        self.activation_first = activation_first

        # normalization layer
        self.normalization = get_normalization_layer(self.channel_axis,
                                                     normalization,
                                                     norm_momentum,
                                                     norm_group)

        # activation layer
        self.activation = get_activation_layer(activation, activation_alpha)

        # padding layer
        self.pad = get_padding_layer(rank=rank,
                                     padding=padding,
                                     pad_type=pad_type,
                                     constant_values=pad_constant_values,
                                     data_format=self.data_format)

        # convolutioni layer
        self.conv = getattr(self, 'conv', None)

    def call(self, inputs):
        outputs = inputs
        # normalization -> activation -> convolution
        if self.normalization_first:
            if self.normalization:
                outputs = self.normalization(outputs)
            if self.activation:
                outputs = self.activation(outputs)
            if self.pad:
                outputs = self.pad(outputs)
            outputs = self.conv(outputs)
        # activation -> convolution -> normalization
        elif self.activation_first:
            if self.activation:
                outputs = self.activation(outputs)
            if self.pad:
                outputs = self.pad(outputs)
            outputs = self.conv(outputs)
            if self.normalization:
                outputs = self.normalization(outputs)
        # convolution -> normalization -> activation
        else:
            if self.pad:
                outputs = self.pad(outputs)
            outputs = self.conv(outputs)
            if self.normalization:
                outputs = self.normalization(outputs)
            if self.activation:
                outputs = self.activation(outputs)
        return outputs

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        return self.rank + 1

    def get_config(self):
        config = {
            'name': self.name,
            'rank': self.rank,
            'normalization_first': self.normalization_first,
            'activation_first': self.activation_first,
            'convolution': get_layer_config(self.conv),
            'normalization': get_layer_config(self.normalization),
            'activation': get_layer_config(self.activation),
            'pad': get_layer_config(self.pad),
        }
        return config


class ConvBlock(BaseBlock):
    """
    Convolution Block.

    Block consists of convolution,
    pad, normalization, and activation layers.
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 conv_padding='valid',
                 dilation_rate=1,
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=rank,
            data_format=data_format,
            padding=padding,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)

        # convolution layer
        self.conv = Conv(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_padding,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=f'conv{rank}d')

        # spectral normalization
        if use_spectral_norm:
            self.conv = SpectralNormalization(
                self.conv,
                power_iterations=spectral_iteration,
                name='spectral_normalization')


class Conv1DBlock(ConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 conv_padding='valid',
                 dilation_rate=1,
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            conv_padding=conv_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_spectral_norm=use_spectral_norm,
            spectral_iteration=spectral_iteration,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)


class Conv2DBlock(ConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 conv_padding='valid',
                 dilation_rate=(1, 1),
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_first=False,
                 activation_alpha=0.3,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            conv_padding=conv_padding,
            dilation_rate=dilation_rate,
            groups=groups,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_spectral_norm=use_spectral_norm,
            spectral_iteration=spectral_iteration,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            data_format=data_format,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)


class Conv3DBlock(ConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=(0, 0, 0),
                 conv_padding='valid',
                 dilation_rate=(1, 1, 1),
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_first=False,
                 activation_alpha=0.3,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            conv_padding=conv_padding,
            dilation_rate=dilation_rate,
            groups=groups,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_spectral_norm=use_spectral_norm,
            spectral_iteration=spectral_iteration,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            data_format=data_format,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)


class TransConvBlock(BaseBlock):
    """
    Transposed Convolution Block.

    Block consists of transposed convolution,
    pad, normalization, and activation layers.
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 conv_padding='valid',
                 output_padding=None,
                 dilation_rate=1,
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=rank,
            data_format=data_format,
            padding=padding,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)

        # convolution layer
        if rank == 1:
            conv = Conv1DTranspose
        elif rank == 2:
            conv = Conv2DTranspose
        else:
            conv = Conv3DTranspose
        self.conv = conv(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_padding,
            output_padding=output_padding,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=f'conv{rank}d_transpose')

        # spectral normalization
        if use_spectral_norm:
            self.conv = SpectralNormalization(
                self.conv,
                power_iterations=spectral_iteration,
                name='spectral_normalization')


class TransConv1DBlock(TransConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 conv_padding='valid',
                 output_padding=None,
                 dilation_rate=1,
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_first=False,
                 activation_alpha=0.3,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            conv_padding=conv_padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_spectral_norm=use_spectral_norm,
            spectral_iteration=spectral_iteration,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)


class TransConv2DBlock(TransConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 conv_padding='valid',
                 output_padding=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_first=False,
                 activation_alpha=0.3,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            conv_padding=conv_padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_spectral_norm=use_spectral_norm,
            spectral_iteration=spectral_iteration,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)


class TransConv3DBlock(TransConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=(0, 0, 0),
                 conv_padding='valid',
                 output_padding=None,
                 dilation_rate=(1, 1, 1),
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_first=False,
                 activation_alpha=0.3,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            conv_padding=conv_padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_spectral_norm=use_spectral_norm,
            spectral_iteration=spectral_iteration,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)


class UpsampleConv2DBlock(BaseBlock):
    """
    Upsample Convolution Block.

    Block consists of upsample convolution,
    pad, normalization, and activation layers.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 conv_padding='valid',
                 size=None,
                 scale=None,
                 method='bilinear',
                 dilation_rate=(1, 1),
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=2,
            data_format=data_format,
            padding=padding,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)

        # convolution layer
        self.conv = UpsampleConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_padding,
            size=size,
            scale=scale,
            method=method,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=f'upsample_conv{self.rank}d')

        # spectral normalization
        if use_spectral_norm:
            self.conv = SpectralNormalization(
                self.conv,
                power_iterations=spectral_iteration,
                name='spectral_normalization')


class SubPixelConv2DBlock(BaseBlock):
    """
    SubPixel Convolution Block.

    Block consists of subpixel convolution,
    pad, normalization, and activation layers.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 conv_padding='valid',
                 scale=2,
                 use_icnr_initializer=False,
                 dilation_rate=(1, 1),
                 groups=1,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=2,
            data_format=data_format,
            padding=padding,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            normalization=normalization,
            normalization_first=normalization_first,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_first=activation_first,
            activation_alpha=activation_alpha,
            trainable=trainable,
            name=name,
            **kwargs)

        # convolution layer
        self.conv = SubPixelConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=conv_padding,
            scale=scale,
            use_icnr_initializer=use_icnr_initializer,
            data_format=self.data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=None,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=f'subpixel_conv{self.rank}d')

        # spectral normalization
        if use_spectral_norm:
            self.conv = SpectralNormalization(
                self.conv,
                power_iterations=spectral_iteration,
                name='spectral_normalization')
