"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import functools
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils


class Resample(tf.keras.layers.Layer):
    def __init__(self,
                 factor=None,
                 size=None,
                 method='nearest',
                 preserve_aspect_ratio=False,
                 antialias=False,
                 mode='up',
                 data_format=None,
                 **kwargs):
        super().__init__(**kwargs)
        if (factor is not None) ^ (size is None):  # XOR operation
            raise ValueError('Either `factor` or `size` should not be None.')
        self.factor = factor
        self.size = size
        self.is_integer_factor = True
        self.method = method.lower()
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.antialias = antialias
        self.mode = self._check_mode(mode)
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.rank = len(input_shape) - 2
        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        assert self.size is not None, 'Resample only works with size argument.'
        self.size = conv_utils.normalize_tuple(self.size, self.rank, 'size')
        self._resize_op = self._get_resize_op()
        self.built = True

    def call(self, inputs):
        return self._resize_op(inputs)

    def _get_resize_op(self):
        # tf.image.resize only supports data format `channels_last`
        if self.rank == 1:
            def _resize_op(inputs):
                outputs = tf.expand_dims(inputs, 2)
                outputs = tf.image.resize(
                    outputs,
                    size=self.size + (1,),
                    method=self.method,
                    preserve_aspect_ratio=self.preserve_aspect_ratio,
                    antialias=self.antialias,
                    name='resize')
                return tf.squeeze(outputs, axis=2)
        elif self.rank == 2:
            _resize_op = functools.partial(
                tf.image.resize,
                size=self.size,
                method=self.method,
                preserve_aspect_ratio=self.preserve_aspect_ratio,
                antialias=self.antialias,
                name='resize')
        else:
            raise NotImplementedError(
                f'Only support nearest method for rank 3.')

        if self.data_format == 'channels_first':
            transpose_axis = (0, *self._spatial_axes, self._channel_axis)
            return_axis = (0, self.rank+1, *range(1, self.rank+1))

            def resize_op(inputs):
                outputs = tf.transpose(inputs, transpose_axis)
                outputs = _resize_op(outputs)
                return tf.transpose(outputs, return_axis)
        else:
            resize_op = _resize_op
        return resize_op

    def _check_mode(self, mode):
        l_mode = mode.lower()
        if l_mode in {'upsample', 'up'}:
            return 'up'
        if l_mode in {'downsample', 'down'}:
            return 'down'
        raise ValueError(f'Unsupported `mode`: {mode}')

    def _check_factored_size(self, input_shape):
        def _check_factor(factor):
            if hasattr(factor, '__len__'):
                if len(factor) != self.rank:
                    raise ValueError('`factor` should have length of `rank`.')
                if self.preserve_aspect_ratio:
                    factor = (min(factor),) * self.rank
            else:
                factor = (factor,) * self.rank
            for f in factor:
                if f < 1:
                    raise ValueError('`factor` should greater than 1.')
                if hasattr(f, 'is_integer') and not f.is_integer():
                    self.is_integer_factor = False
                    return factor
            return tuple(map(int, factor))

        if self.size is None:
            self.factor = _check_factor(self.factor)
            if self.mode == 'up':
                self.size = tuple(
                    int(input_shape[axis] * factor)
                    for axis, factor in zip(self._spatial_axes, self.factor))
            else:
                self.size = tuple(
                    input_shape[axis] // factor
                    for axis, factor in zip(self._spatial_axes, self.factor))
        else:
            self.size = conv_utils.normalize_tuple(
                self.size, self.rank, 'size')
            if self.mode == 'up':
                self.factor = _check_factor(
                    size / input_shape[axis]
                    for axis, size in zip(self._spatial_axes, self.size))
            else:
                self.factor = _check_factor(
                    input_shape[axis] / size
                    for axis, size in zip(self._spatial_axes, self.size))

        if self.rank == 3 and not self.is_integer_factor:
            raise ValueError(
                f'`factor` is not integer. Received: {self.factor}')

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

    def get_config(self):
        config = super().get_config()
        config.update({
            'factor': self.factor,
            'method': self.method,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'antialias': self.antialias
        })


class Upsample(Resample):
    def __init__(self,
                 factor=None,
                 size=None,
                 method='nearest',
                 preserve_aspect_ratio=False,
                 antialias=False,
                 data_format=None,
                 **kwargs):
        super().__init__(
            factor=factor,
            size=size,
            method=method,
            preserve_aspect_ratio=preserve_aspect_ratio,
            antialias=antialias,
            data_format=data_format,
            **kwargs)
        if factor is not None and factor <= 1:
            raise ValueError('`factor` should greater than 1.')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.rank = len(input_shape) - 2
        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        self._check_factored_size(input_shape)

        if not self.is_integer_factor:
            self._resize_op = self._get_resize_op()
            self.built = True
            return

        input_channels = int(input_shape[self._channel_axis])
        new_shape = []
        tile_multiples = []
        pad_multiples = []
        return_shape = []
        for axis, factor in zip(self._spatial_axes, self.factor):
            new_shape.extend([input_shape[axis], 1])
            tile_multiples.extend([1, factor])
            pad_multiples.extend([(0, 0), (0, factor-1)])
            return_shape.append(input_shape[axis] * factor)
        if self.data_format == 'channels_first':
            new_shape = (-1, input_channels, *new_shape)
            tile_multiples = (1, 1, *tile_multiples)
            pad_multiples = ((0, 0), (0, 0), *pad_multiples)
            return_shape = (-1, input_channels, *return_shape)
        else:
            new_shape = (-1, *new_shape, input_channels)
            tile_multiples = (1, *tile_multiples, 1)
            pad_multiples = ((0, 0), *pad_multiples, (0, 0))
            return_shape = (-1, *return_shape, input_channels)

        if self.method == 'nearest':
            def resize_op(inputs):
                outputs = tf.reshape(inputs, new_shape)
                outputs = tf.tile(outputs, tile_multiples)
                return tf.reshape(outputs, return_shape)
        elif self.method.startswith('zero'):
            def resize_op(inputs):
                outputs = tf.reshape(inputs, new_shape)
                outputs = tf.pad(outputs, pad_multiples,
                                 mode='CONSTANT', constant_values=0)
                return tf.reshape(outputs, return_shape)

        self._resize_op = resize_op
        self.built = True


class Downsample(Resample):
    def __init__(self,
                 factor=None,
                 size=None,
                 method='nearest',
                 preserve_aspect_ratio=False,
                 antialias=False,
                 data_format=None,
                 **kwargs):
        super().__init__(
            factor=factor,
            size=size,
            method=method,
            preserve_aspect_ratio=preserve_aspect_ratio,
            antialias=antialias,
            data_format=data_format,
            **kwargs)
        if factor is not None and factor <= 1:
            raise ValueError('`factor` should greater than 1.')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.rank = len(input_shape) - 2
        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        self._check_factored_size(input_shape)

        input_channels = int(input_shape[self._channel_axis])
        if self.method == 'nearest' and self.is_integer_factor:
            new_shape = []
            reduction_axes = []
            for i, (axis, factor) in enumerate(zip(self._spatial_axes,
                                                   self.factor)):
                if input_shape[axis] % factor:
                    break
                new_shape.extend([input_shape[axis] // factor, factor])
                reduction_axes.append(axis + i + 1)
            else:
                if self.data_format == 'channels_first':
                    new_shape = (-1, input_channels, *new_shape)
                else:
                    new_shape = (-1, *new_shape, input_channels)

                def resize_op(inputs):
                    outputs = tf.reshape(inputs, new_shape)
                    return tf.reduce_mean(outputs, axis=reduction_axes)
                self._resize_op = resize_op
                self.built = True
                return

            def resize_op(inputs):
                if self.data_format == 'channels_first':
                    if self.rank == 1:
                        return inputs[..., ::self.factor[0]]
                    if self.rank == 2:
                        return inputs[..., ::self.factor[0], ::self.factor[1]]
                    if self.rank == 3:
                        return inputs[..., ::self.factor[0], ::self.factor[1], ::self.factor[2]]
                elif self.data_format == 'channels_last':
                    if self.rank == 1:
                        return inputs[..., ::self.factor[0], :]
                    if self.rank == 2:
                        return inputs[..., ::self.factor[0], ::self.factor[1], :]
                    if self.rank == 3:
                        return inputs[..., ::self.factor[0], ::self.factor[1], ::self.factor[2], :]
        else:
            resize_op = self._get_resize_op()
        self._resize_op = resize_op
        self.built = True
