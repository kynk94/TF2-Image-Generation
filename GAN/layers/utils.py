"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization
from .normalizations import FilterResponseNormalization
from .padding import Padding


def get_layer_config(layer):
    if layer is None:
        return None
    if hasattr(layer, 'get_config'):
        return layer.get_config()
    return getattr(layer, '__name__',
                   layer.__class__.__name__)


def get_padding_layer(rank, padding, pad_type, constant_values, data_format):
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
        normalization = normalization.lower()
        if normalization in {'batch_normalization',
                             'batch_norm', 'bn'}:
            return BatchNormalization(
                axis=channel_axis,
                momentum=normalization_momentum,
                epsilon=normalization_epsilon,
                name='batch_normalization')
        if normalization in {'layer_normalization',
                             'layer_norm', 'ln'}:
            return LayerNormalization(
                axis=channel_axis,
                epsilon=normalization_epsilon,
                name='layer_normalization')
        if normalization in {'instance_normalization',
                             'instance_norm', 'in'}:
            return InstanceNormalization(
                axis=channel_axis,
                epsilon=normalization_epsilon,
                name='instance_normalization')
        if normalization in {'group_normalization',
                             'group_norm', 'gn'}:
            return GroupNormalization(
                axis=channel_axis,
                groups=normalization_group,
                epsilon=normalization_epsilon,
                name='group_normalization')
        if normalization in {'filter_response_normalization',
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
        activation = activation.lower()
        if activation == 'relu':
            return tf.keras.layers.ReLU(name='relu')
        if activation in {'leaky_relu', 'lrelu'}:
            return tf.keras.layers.LeakyReLU(
                alpha=activation_alpha,
                name='leaky_relu')
        if activation in {'exp_lu', 'elu'}:
            return tf.keras.layers.ELU(
                alpha=activation_alpha,
                name='elu')
        if activation in {'trelu', 'tlu'}:
            return tfa.layers.TLU()
        if activation == 'tanh':
            return tf.nn.tanh
    raise ValueError(f'Unsupported `activation`: {activation}')


def kwargs_as_iterable(iter_len, **kwargs):
    for key, value in kwargs.items():
        if hasattr(value, '__len__'):
            if len(value) == iter_len:
                continue
            raise ValueError(f'`len({key})` does not match `iter_len`.')
        kwargs[key] = (value, ) * iter_len
    return kwargs
