"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).

wavelet transform for SWAGAN (https://arxiv.org/abs/2102.06108)
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from .filters import FIRFilter
from .padding import Padding
from .resample import Downsample, Upsample


class HaarTransform2D(tf.keras.layers.Layer):
    def __init__(self,
                 concat_direction='channel',
                 data_format=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.concat_direction = self.get_concat_direction(concat_direction)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.transforms = [FIRFilter(kernel=[[1, 1],
                                             [1, 1]],
                                     padding=0,
                                     kernel_normalize=True),
                           FIRFilter(kernel=[[-1, -1],
                                             [1, 1]],
                                     padding=0,
                                     kernel_normalize=True),
                           FIRFilter(kernel=[[-1, 1],
                                             [-1, 1]],
                                     padding=0,
                                     kernel_normalize=True),
                           FIRFilter(kernel=[[1, -1],
                                             [-1, 1]],
                                     padding=0,
                                     kernel_normalize=True)]
        self.resample_layer = Downsample(factor=2, method='nearest')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.rank = len(input_shape) - 2
        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        for transform in self.transforms:
            transform.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        outputs = [self.resample_layer(transform(inputs))
                   for transform in self.transforms]
        if self.concat_direction == 'spatial':
            return tf.concat([
                tf.concat(outputs[:2], axis=self._spatial_axes[1]),
                tf.concat(outputs[2:], axis=self._spatial_axes[1])
            ], axis=self._spatial_axes[0])
        return tf.concat(outputs, axis=self._channel_axis)

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

    def get_concat_direction(self, concat_direction: str):
        l_concat_direction = concat_direction.lower()
        if 'spatial' in l_concat_direction:
            return 'spatial'
        if 'channel' in l_concat_direction:
            return 'channel'


class HaarInverseTransform2D(HaarTransform2D):
    def __init__(self,
                 concat_direction='channel',
                 data_format=None,
                 **kwargs):
        super().__init__(concat_direction=concat_direction,
                         data_format=data_format,
                         **kwargs)
        for i in (1, 2):
            self.transforms[i].kernel = -np.array(self.transforms[i].kernel)
        self.padding_layer = Padding(rank=2, padding=((1, 0), (1, 0)))
        self.resample_layer = Upsample(factor=2, method='nearest')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.rank = len(input_shape) - 2
        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        if self.concat_direction == 'spatial':
            for spatial_axis in self._spatial_axes:
                input_shape.dims[spatial_axis] //= 2
        else:
            input_shape.dims[self._channel_axis] //= 4
        super().build(input_shape)

    def call(self, inputs):
        if self.concat_direction == 'spatial':
            temporal_inputs = tf.split(inputs, 2, axis=self._spatial_axes[0])
            inputs = []
            for temporal_input in temporal_inputs:
                inputs.extend(tf.split(temporal_input, 2,
                                       axis=self._spatial_axes[1]))
        else:
            inputs = tf.split(inputs, 4, axis=self._channel_axis)

        outputs = 0.0
        for input, transform in zip(inputs, self.transforms):
            outputs += transform(self.padding_layer(self.resample_layer(input)))
        return outputs
