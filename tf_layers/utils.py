"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import tensorflow as tf
import tensorflow_addons as tfa
from .noise import GaussianNoise, UniformNoise
from .padding import Padding


def hasattr_not_none(__obj, attr):
    if not hasattr(__obj, attr):
        return False
    if getattr(__obj, attr) is None:
        return False
    return True


def kwargs_as_iterable(iter_len, **kwargs):
    for key, value in kwargs.items():
        if not isinstance(value, str) and hasattr(value, '__len__'):
            if len(value) == iter_len:
                continue
            raise ValueError(f'`len({key})` does not match `iter_len`.')
        kwargs[key] = (value, ) * iter_len
    return kwargs


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
        if l_noise.startswith('uniform'):
            return UniformNoise(
                strength=strength,
                channel_same=channel_same,
                trainable=trainable,
                **kwargs)
    raise ValueError(f'Unsupported `noise`: {noise}')


def get_activation_layer(activation, activation_alpha=0.3):
    if activation is None:
        return None
    if hasattr(activation, '__call__'):
        return activation
    if not isinstance(activation, str):
        raise ValueError(f'Unsupported `activation`: {activation}')
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
    return tf.keras.layers.Activation(l_activation)
