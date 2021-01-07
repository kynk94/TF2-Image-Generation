"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization
from .filters import FIRFilter
from .noise import GaussianNoise, UniformNoise
from .normalizations import FilterResponseNormalization
from .padding import Padding


def check_tf_version():
    version = []
    for string in tf.__version__.split('.'):
        version.append(int(''.join(s for s in string if s.isdigit())))
    assert version[0] == 2, 'Only support tensorflow 2.x'
    return version


def get_layer_config(layer):
    if layer is None:
        return None
    if hasattr(layer, 'get_config'):
        return layer.get_config()
    return getattr(layer, '__name__',
                   layer.__class__.__name__)


def get_initializer(initializer, use_weight_scaling=False, lr_multiplier=1.0):
    if not use_weight_scaling:
        return tf.keras.initializers.get(initializer)
    stddev = 1.0 / lr_multiplier
    return tf.initializers.random_normal(0, stddev)


def get_str_padding(padding):
    if isinstance(padding, str):
        l_padding = padding.lower()
        if l_padding in {'same', 'valid'}:
            return l_padding
        raise ValueError(f'Unsupported `padding`: {padding}')
    return 'valid'


def get_padding_layer(rank, padding, pad_type, constant_values, data_format):
    if isinstance(padding, str):
        return None

    def _check_padding(rank, padding):
        if hasattr(padding, '__len__'):
            if len(padding) != rank:
                raise ValueError('`padding` should have `rank` elements. '
                                 f'Found: {len(padding)}')

            def check_all_zero(value):
                if hasattr(value, '__len__'):
                    if len(value) == 0:
                        return False
                    return all(check_all_zero(v) for v in value)
                return value == 0

            if check_all_zero(padding):
                return 0
            return padding
        if padding >= 0:
            return (padding, ) * rank
        raise ValueError(f'Unsupported `padding`: {padding}')

    padding = _check_padding(rank, padding)
    if padding == 0:
        return None
    return Padding(rank=rank,
                   padding=padding,
                   pad_type=pad_type,
                   constant_values=constant_values,
                   data_format=data_format,
                   name=f'padding{rank}d')


def get_noise_layer(noise,
                    strength=0.0,
                    channel_same=True,
                    trainable=True,
                    **kwargs):
    if noise is None:
        return None
    if hasattr(noise, '__call__'):
        return noise
    if isinstance(noise, str):
        l_noise = noise.lower()
        if l_noise in {'gaussian_noise', 'gaussian', 'normal_noise', 'normal'}:
            return GaussianNoise(
                strength=strength,
                channel_same=channel_same,
                trainable=trainable,
                **kwargs)
        if l_noise in {'uniform_noise', 'uniform'}:
            return UniformNoise(
                strength=strength,
                channel_same=channel_same,
                trainable=trainable,
                **kwargs)


def get_filter_layer(filter,
                     factor=2,
                     gain=1,
                     stride=1,
                     kernel_normalize=True,
                     data_format=None):
    if filter is None:
        return None
    if filter == True:
        return FIRFilter(factor=factor,
                         gain=gain,
                         stride=stride,
                         kernel_normalize=kernel_normalize,
                         data_format=data_format)
    return FIRFilter(kernel=filter,
                     factor=factor,
                     gain=gain,
                     stride=stride,
                     kernel_normalize=kernel_normalize,
                     data_format=data_format)


def get_normalization_layer(channel_axis,
                            normalization,
                            normalization_momentum=0.99,
                            normalization_group=32,
                            normalization_epsilon=1e-5):
    if normalization is None:
        return None
    if hasattr(normalization, '__call__'):
        return normalization
    if isinstance(normalization, str):
        l_normalization = normalization.lower()
        if l_normalization in {'batch_normalization',
                               'batch_norm', 'bn'}:
            return BatchNormalization(
                axis=channel_axis,
                momentum=normalization_momentum,
                epsilon=normalization_epsilon,
                name='batch_normalization')
        if l_normalization in {'layer_normalization',
                               'layer_norm', 'ln'}:
            return LayerNormalization(
                axis=channel_axis,
                epsilon=normalization_epsilon,
                name='layer_normalization')
        if l_normalization in {'instance_normalization',
                               'instance_norm', 'in'}:
            return InstanceNormalization(
                axis=channel_axis,
                epsilon=normalization_epsilon,
                name='instance_normalization')
        if l_normalization in {'group_normalization',
                               'group_norm', 'gn'}:
            return GroupNormalization(
                axis=channel_axis,
                groups=normalization_group,
                epsilon=normalization_epsilon,
                name='group_normalization')
        if l_normalization in {'filter_response_normalization',
                               'filter_response_norm', 'frn'}:
            # FilterResponseNormalization is not official implementation.
            # Official need to input axis as spatial, not channel.
            return FilterResponseNormalization(
                axis=channel_axis,
                epsilon=normalization_epsilon,
                name='filter_response_normalization')
    raise ValueError(f'Unsupported `normalization`: {normalization}')


def get_activation_layer(activation, activation_alpha=0.3):
    if activation is None:
        return None
    if hasattr(activation, '__call__'):
        return activation
    if isinstance(activation, str):
        l_activation = activation.lower()
        if l_activation == 'relu':
            return tf.keras.layers.ReLU(name='relu')
        if l_activation in {'leaky_relu', 'lrelu'}:
            return tf.keras.layers.LeakyReLU(
                alpha=activation_alpha,
                name='leaky_relu')
        if l_activation in {'exp_lu', 'elu'}:
            return tf.keras.layers.ELU(
                alpha=activation_alpha,
                name='elu')
        if l_activation in {'trelu', 'tlu'}:
            return tfa.layers.TLU()
        if l_activation == 'tanh':
            return tf.keras.layers.Activation('tanh')
    raise ValueError(f'Unsupported `activation`: {activation}')


def kwargs_as_iterable(iter_len, **kwargs):
    for key, value in kwargs.items():
        if not isinstance(value, str) and hasattr(value, '__len__'):
            if len(value) == iter_len:
                continue
            raise ValueError(f'`len({key})` does not match `iter_len`.')
        kwargs[key] = (value, ) * iter_len
    return kwargs
