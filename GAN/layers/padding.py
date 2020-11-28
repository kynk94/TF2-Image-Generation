"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import functools
import tensorflow as tf
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.keras.utils.conv_utils import normalize_data_format, normalize_tuple


class Padding(tf.keras.layers.Layer):
    """
    Padding layer for N rank input.
    `rank` equivalent to `len(tensor.shape) - 2`

    Arguments:
        rank: An integer, the rank of the input `tensor`.
        padding: Int, or tuple of int (length N).
            - If int:
                the same symmetric padding
                is applied to N spatial dimension.
            - If tuple of N ints:
                interpreted as two different symmetric padding
                values for N spatial dimension.
            - If tuple of N tuples of 2 ints:
                interpreted as
                `((left_dim1_pad, right_dim1_pad),
                  ...,
                  (left_dimN_pad, right_dimN_pad))`
        pad_type: A string,
            one of {"constant", "zero", "reflect", "symmetric"}.
        constant_values: In "CONSTANT" mode, the scalar pad value to use.
            Must be same type as `tensor`.
        data_format: A string,
            one of {"channels_last", "channels_first"}.
    """

    def __init__(self,
                 rank,
                 padding=1,
                 pad_type='constant',
                 constant_values=0,
                 data_format=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.pad_type, self.constant_values = self._check_pad_params(
            pad_type, constant_values)
        self.constant_values = constant_values
        self.data_format = normalize_data_format(data_format)

        if isinstance(padding, int):
            self.padding = ((padding,)*2,) * rank
        elif hasattr(padding, '__len__'):
            if len(padding) != rank:
                raise ValueError(f'`padding` should have {rank} elements. '
                                 f'Found: {padding}')
            dim_padding = []
            for i in range(rank):
                dim_padding.append(normalize_tuple(padding[i], 2,
                                                   f'{i} entry of padding'))
            self.padding = tuple(dim_padding)
        else:
            raise ValueError(
                '`padding` should be either an int, '
                f'a tuple of {rank} ints '
                '(' + ', '.join(
                    f'symmetric_dim{i}_pad'
                    for i in range(1, rank+1)) + '), '
                f'or a tuple of {rank} tuples of 2 ints '
                '(' + ', '.join(
                    f'(left_dim{i}_pad, right_dim{i}_pad)'
                    for i in range(1, rank+1)) + '). '
                f'Found: {padding}')

        self.padding = ((0, 0),) + self.padding
        if self.data_format == 'channels_first':
            self.padding = ((0, 0),) + self.padding
        else:
            self.padding += ((0, 0),)

    def _check_pad_params(self, pad_type, constant_values):
        if not isinstance(constant_values, (int, float)):
            raise ValueError(f'`constant_values` should be either int or float.'
                             f'Found: {constant_values}')
        pad_type_upper = pad_type.upper()
        if pad_type_upper in {'CONSTANT', 'REFLECT', 'SYMMETRIC'}:
            return pad_type_upper, constant_values
        if pad_type_upper == 'ZERO':
            return 'CONSTANT', 0
        raise ValueError(f'Unsupported `pad_type`: {pad_type}')

    def build(self, input_shape):
        tf_op_name = self.__class__.__name__

        self._pad_op = functools.partial(
            tf.pad,
            paddings=self.padding,
            mode=self.pad_type,
            constant_values=self.constant_values,
            name=tf_op_name)
        self.built = True

    def call(self, inputs):
        return self._pad_op(inputs)

    def compute_output_shape(self, input_shape):
        input_shape = TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            batch, channel, *dims = input_shape
            padding_index = 0
        else:
            batch, *dims, channel = input_shape
            padding_index = 1
        output_dims = []
        for i, dim in enumerate(dims):
            if dim is None:
                output_dims.append(None)
            else:
                dim += 2 * self.padding[i][padding_index]
                output_dims.append(dim)
        if self.data_format == 'channels_first':
            return TensorShape([batch, channel, *output_dims])
        return TensorShape([batch, *output_dims, channel])

    def get_config(self):
        config = super().get_config()
        config.update({
            'rank':
                self.rank,
            'padding':
                self.padding,
            'pad_type':
                self.pad_type,
            'constant_values':
                self.constant_values,
            'data_format':
                self.data_format
        })
        return config


class Padding1D(Padding):
    def __init__(self,
                 padding=1,
                 pad_type='constant',
                 constant_values=0,
                 data_format=None,
                 **kwargs):
        super().__init__(
            rank=1,
            padding=padding,
            pad_type=pad_type,
            constant_values=constant_values,
            data_format=data_format,
            **kwargs)


class Padding2D(Padding):
    def __init__(self,
                 padding=1,
                 pad_type='constant',
                 constant_values=0,
                 data_format=None,
                 **kwargs):
        super().__init__(
            rank=2,
            padding=padding,
            pad_type=pad_type,
            constant_values=constant_values,
            data_format=data_format,
            **kwargs)


class Padding3D(Padding):
    def __init__(self,
                 padding=1,
                 pad_type='constant',
                 constant_values=0,
                 data_format=None,
                 **kwargs):
        super().__init__(
            rank=3,
            padding=padding,
            pad_type=pad_type,
            constant_values=constant_values,
            data_format=data_format,
            **kwargs)
