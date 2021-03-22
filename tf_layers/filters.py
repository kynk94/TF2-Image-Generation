import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from .padding import Padding


class FIRFilter(tf.keras.layers.Layer):
    def __init__(self,
                 kernel=None,
                 factor=2,
                 gain=1,
                 stride=1,
                 padding='same',
                 kernel_normalize=True,
                 data_format=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.factor = factor
        self.gain = gain
        self.stride = stride
        self.kernel_normalize = kernel_normalize
        self.padding = padding
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        self.rank = len(input_shape) - 2

        self.kernel = self._setup_kernel()
        flipped_kernel = np.flip(self.kernel)
        kernel_shape = (*self.kernel.shape, 1, 1)
        kernel = self.kernel.reshape(kernel_shape)
        flipped_kernel = flipped_kernel.reshape(kernel_shape)

        kernel = tf.constant(kernel,
                             dtype=self.dtype,
                             name='kernel')
        flipped_kernel = tf.constant(flipped_kernel,
                                     dtype=self.dtype,
                                     name='flipped_kernel')

        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        self.input_channel = input_shape[self._channel_axis]
        strides = conv_utils.normalize_tuple(self.stride, self.rank, 'stride')

        # forward padding
        if isinstance(self.padding, str) and self.padding.lower() == 'same':
            padding_dims = []
            for k in kernel_shape[:2]:
                div, mod = divmod(k - 1, 2)
                padding_dims.append((div, div + mod))
        else:
            padding_dims = self.padding
        self._pad = Padding(rank=self.rank, padding=padding_dims,
                            pad_type='constant', constant_values=0,
                            data_format=self.data_format)
        padded_dims = [input_shape[axis] + sum(self._pad.padding[axis])
                       for axis in self._spatial_axes]
        output_dims = [(padded_dim - k) // stride + 1
                       for padded_dim, k, stride in zip(padded_dims,
                                                        kernel_shape,
                                                        strides)]
        # backprop padding
        back_padding_dims = []
        for i, axis in enumerate(self._spatial_axes):
            k = kernel_shape[i]
            pad = self._pad.padding[axis]
            back_padding_dims.append((k - pad[0] - 1, k - pad[1] - 1))
        self._back_pad = Padding(rank=self.rank, padding=back_padding_dims,
                                 pad_type='constant', constant_values=0,
                                 data_format=self.data_format)
        back_padded_dims = [output_dim + sum(self._back_pad.padding[axis])
                            for output_dim, axis in zip(output_dims, self._spatial_axes)]
        back_output_dims = [(padded_dim - k) // stride + 1
                            for padded_dim, k, stride in zip(back_padded_dims,
                                                             kernel_shape,
                                                             strides)]

        _tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        if self.data_format == 'channels_first':
            def reshape_conv_op(inputs, new_shape, return_shape, padding, filters, name):
                outputs = tf.pad(inputs, padding)
                outputs = tf.reshape(outputs, new_shape)
                outputs = tf.nn.convolution(outputs,
                                            filters=filters,
                                            strides=list(strides),
                                            padding='VALID',
                                            dilations=[1] * self.rank,
                                            data_format=_tf_data_format,
                                            name=name)
                return tf.reshape(outputs, return_shape)
        else:
            transpose_axes = (0, self._channel_axis, *range(1, self.rank+1))
            return_axes = (0, *range(2, self.rank+2), 1)

            def reshape_conv_op(inputs, new_shape, return_shape, padding, filters, name):
                outputs = tf.pad(inputs, padding)
                outputs = tf.transpose(outputs, transpose_axes)
                outputs = tf.reshape(outputs, new_shape)
                outputs = tf.transpose(outputs, return_axes)
                outputs = tf.nn.convolution(outputs,
                                            filters=filters,
                                            strides=list(strides),
                                            padding='VALID',
                                            dilations=[1] * self.rank,
                                            data_format=_tf_data_format,
                                            name=name)
                outputs = tf.transpose(outputs, transpose_axes)
                outputs = tf.reshape(outputs, return_shape)
                return tf.transpose(outputs, return_axes)

        self._conv_op = functools.partial(
            reshape_conv_op,
            padding=self._pad.padding,
            new_shape=(-1, 1, *padded_dims),
            return_shape=(-1, self.input_channel, *output_dims),
            filters=kernel,
            name='fir_forward')
        self._back_conv_op = functools.partial(
            reshape_conv_op,
            padding=self._back_pad.padding,
            new_shape=(-1, 1, *back_padded_dims),
            return_shape=(-1, self.input_channel, *back_output_dims),
            filters=flipped_kernel,
            name='fir_backward')

    @tf.custom_gradient
    def call(self, inputs):
        outputs = self._conv_op(inputs)

        @tf.custom_gradient
        def grad(d_outputs):
            d_inputs = self._back_conv_op(d_outputs)
            return d_inputs, lambda dd_inputs: self._conv_op(dd_inputs)
        return outputs, grad

    def _setup_kernel(self):
        kernel = self.kernel
        if kernel is None:
            kernel = [1] * self.factor
        elif isinstance(kernel, int):
            kernel = [kernel] * self.factor

        kernel = np.array(kernel, np.float32)

        if kernel.ndim == self.rank:
            if self.kernel_normalize:
                kernel /= np.sum(np.abs(kernel))
            return kernel * self.gain
        elif kernel.ndim != 1:
            raise ValueError(f'`kernel` should have dimensioin {self.rank}. '
                             f'Received: {kernel.ndim}')

        kernels = []
        for i in range(self.rank):
            shape = [1] * self.rank
            shape[i] = -1
            kernels.append(kernel.reshape(shape))
        kernel = np.prod(kernels)

        if self.kernel_normalize:
            kernel /= np.sum(np.abs(kernel))
        return kernel * self.gain

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

    def _spatial_output_shape(self, spatial_input_shape):
        spatial_output_shape = [
            conv_utils.conv_output_length(
                length,
                self.kernel.shape[i],
                padding='valid',
                stride=self.stride,
                dilation=1)
            for i, length in enumerate(spatial_input_shape)
        ]

        # spatial output shape for custom pad
        padding = self._pad.padding
        return [
            length + sum(padding[self._spatial_axes[i]]) * self.stride
            for i, length in enumerate(spatial_output_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tf.TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.input_channel])
        else:
            return tf.TensorShape(
                input_shape[:batch_rank] + [self.input_channel] +
                self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel': self.kernel,
            'stride': self.stride,
            'data_format': self.data_format
        })
        return config
