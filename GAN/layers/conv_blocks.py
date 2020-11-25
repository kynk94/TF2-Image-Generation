import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.python.keras.utils.conv_utils import normalize_data_format
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization, FilterResponseNormalization
from tensorflow_addons.layers import SpectralNormalization
from .conv import Conv, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from .padding import Padding
from .utils import get_layer_config


class ConvBlock(tf.keras.Model):
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 pad_type='constant',
                 pad_constant_values=0,
                 conv_padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 normalization=None,
                 norm_group=5,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
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
                 mode='down',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.data_format = normalize_data_format(data_format)
        self.channel_axis = self._get_channel_axis()
        self.activation_first = activation_first

        # padding layer
        self.padding = self._check_padding(padding)
        if self.padding == 0:
            self.pad = None
        else:
            self.pad = Padding(rank=self.rank,
                               padding=self.padding,
                               pad_type=pad_type,
                               constant_values=pad_constant_values,
                               data_format=self.data_format,
                               name=f'padding{rank}d')

        # convolution layer
        if mode.lower() in {'downsample', 'down'}:
            self.mode = 'downsample'
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
        elif mode.lower() in {'upsample', 'up'}:
            self.mode = 'upsample'
            if rank == 1:
                conv_transpose = Conv1DTranspose
            elif rank == 2:
                conv_transpose = Conv2DTranspose
            else:
                conv_transpose = Conv3DTranspose
            self.conv = conv_transpose(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=conv_padding,
                output_padding=output_padding,
                data_format=self.data_format,
                dilation_rate=dilation_rate,
                groups=groups,
                activation=None,
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
        else:
            raise ValueError(f'Unsupported `mode`: {mode}')

        # spectral normalization
        if use_spectral_norm:
            self.conv = SpectralNormalization(self.conv,
                                              power_iterations=spectral_iteration,
                                              name='spectral_normalization')

        # normalization layer
        if normalization is None:
            self.normalization = None
        elif hasattr(normalization, '__call__'):
            self.normalization = normalization
        elif isinstance(normalization, str):
            normalization = normalization.lower()
            if normalization in {'batch_normalization',
                                 'batch_norm', 'bn'}:
                self.normalization = BatchNormalization(
                    axis=self.channel_axis,
                    name='batch_normalization')
            elif normalization in {'layer_normalization',
                                   'layer_norm', 'ln'}:
                self.normalization = LayerNormalization(
                    axis=self.channel_axis,
                    name='layer_normalization')
            elif normalization in {'instance_normalization',
                                   'instance_norm', 'in'}:
                self.normalization = InstanceNormalization(
                    axis=self.channel_axis,
                    name='instance_normalization')
            elif normalization in {'group_normalization',
                                   'group_norm', 'gn'}:
                self.normalization = GroupNormalization(
                    groups=norm_group,
                    axis=self.channel_axis,
                    name='group_normalization')
            elif normalization in {'filter_response_normalization',
                                   'filter_response_norm', 'frn'}:
                axis = list(range(1, self.rank))
                del axis[self.channel_axis]
                self.normalization = FilterResponseNormalization(
                    axis=axis,
                    name='filter_response_normalization')
            else:
                raise ValueError(
                    f'Unsupported `normalization`: {normalization}')
        else:
            raise ValueError(f'Unsupported `normalization`: {normalization}')

        # activation layer
        if activation is None:
            self.activation = None
        elif hasattr(activation, '__call__'):
            self.activation = activation
        elif isinstance(activation, str):
            activation = activation.lower()
            if activation == 'relu':
                self.activation = tf.keras.layers.ReLU(name='relu')
            elif activation in {'leaky_relu', 'lrelu'}:
                self.activation = tf.keras.layers.LeakyReLU(
                    alpha=activation_alpha,
                    name='leaky_relu')
            elif activation in {'exp_lu', 'elu'}:
                self.activation = tf.keras.layers.ELU(
                    alpha=activation_alpha,
                    name='elu')
            else:
                raise ValueError(f'Unsupported `activation`: {activation}')
        else:
            raise ValueError(f'Unsupported `activation`: {activation}')

    def _check_padding(self, padding):
        if hasattr(padding, '__len__'):
            def check_all_zero(value):
                if hasattr(value, '__len__'):
                    if len(value) == 0:
                        return False
                    return all(check_all_zero(v) for v in value)
                return value == 0
            if check_all_zero(padding):
                return 0
            return padding
        elif padding >= 0:
            return padding
        raise ValueError(f'Unsupported `padding`: {padding}')

    def call(self, inputs):
        outputs = inputs
        if self.activation_first:
            if self.activation:
                outputs = self.activation(outputs)
            if self.pad:
                outputs = self.pad(outputs)
            outputs = self.conv(outputs)
            if self.normalization:
                outputs = self.normalization(outputs)
            return outputs

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
            'mode': self.mode,
            'activation_first': self.activation_first,
            'activation': get_layer_config(self.activation),
            'convolution': get_layer_config(self.conv),
            'normalization': get_layer_config(self.normalization),
            'pad': get_layer_config(self.pad),
        }
        return config


class Conv1DBlock(ConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 pad_type='constant',
                 pad_constant_values=0,
                 conv_padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 normalization=None,
                 norm_group=5,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
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
                 mode='down',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            conv_padding=conv_padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            normalization=normalization,
            norm_group=norm_group,
            activation=activation,
            activation_alpha=activation_alpha,
            activation_first=activation_first,
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
            mode=mode,
            trainable=trainable,
            name=name,
            **kwargs)


class Conv2DBlock(ConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=0,
                 pad_type='constant',
                 pad_constant_values=0,
                 conv_padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 normalization=None,
                 norm_group=5,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
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
                 mode='down',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            conv_padding=conv_padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            normalization=normalization,
            norm_group=norm_group,
            activation=activation,
            activation_alpha=activation_alpha,
            activation_first=activation_first,
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
            mode=mode,
            trainable=trainable,
            name=name,
            **kwargs)


class Conv3DBlock(ConvBlock):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=0,
                 pad_type='constant',
                 pad_constant_values=0,
                 conv_padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 normalization=None,
                 norm_group=5,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
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
                 mode='down',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            conv_padding=conv_padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            normalization=normalization,
            norm_group=norm_group,
            activation=activation,
            activation_alpha=activation_alpha,
            activation_first=activation_first,
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
            mode=mode,
            trainable=trainable,
            name=name,
            **kwargs)
