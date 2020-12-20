import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils


class FIRFilter(tf.keras.layers.Layer):
    def __init__(self,
                 kernel=None,
                 factor=2,
                 gain=1,
                 stride=1,
                 kernel_normalize=True,
                 data_format=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel
        self.factor = factor
        self.gain = gain
        self.stride = stride
        self.kernel_normalize = kernel_normalize
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        self.rank = len(input_shape) - 2
        channel_axis = self._get_channel_axis()
        input_channel = input_shape[channel_axis]
        spatial_dims = [input_shape[axis] for axis in self._get_spatial_axes()]
        new_shape = (-1, 1, *spatial_dims)
        return_shape = (-1, input_channel, *spatial_dims)

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

        strides = conv_utils.normalize_tuple(self.stride, self.rank, 'stride')
        _tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)

        if self.data_format == 'channels_first':
            def reshape_conv_op(inputs, filters, name):
                outputs = tf.reshape(inputs, new_shape)
                outputs = tf.nn.convolution(outputs,
                                            filters=filters,
                                            strides=list(strides),
                                            padding='SAME',
                                            dilations=[1] * self.rank,
                                            data_format=_tf_data_format,
                                            name=name)
                return tf.reshape(outputs, return_shape)
        else:
            transpose_axes = (0, channel_axis, *range(1, self.rank+1))
            return_axes = (0, *range(2, self.rank+2), 1)

            def reshape_conv_op(inputs, filters, name):
                outputs = tf.transpose(inputs, transpose_axes)
                outputs = tf.reshape(outputs, new_shape)
                outputs = tf.nn.convolution(outputs,
                                            filters=filters,
                                            strides=list(strides),
                                            padding='SAME',
                                            dilations=[1] * self.rank,
                                            data_format=_tf_data_format,
                                            name=name)
                outputs = tf.reshape(outputs, return_shape)
                return tf.transpose(return_axes)

        self._conv_op = functools.partial(
            reshape_conv_op,
            filters=kernel,
            name='fir_forward')
        self._back_conv_op = functools.partial(
            reshape_conv_op,
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
                kernel /= np.sum(kernel)
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
            kernel /= np.sum(kernel)
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

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel': self.kernel,
            'stride': self.stride,
            'data_format': self.data_format
        })
        return config
