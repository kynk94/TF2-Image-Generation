"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
from .ICNR_initializer import ICNR
from .resample import Downsample, Upsample
from .utils import get_filter_layer, get_padding_layer, get_noise_layer
from .utils import get_initializer, get_layer_config, get_str_padding


class Conv(convolutional.Conv):
    """
    Inherited from the official tf implementation.
    (edited by https://github.com/kynk94)

    Abstract N-D convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Arguments:
        rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space (i.e. the number
            of filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            length of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the stride length of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: An integer or tuple/list of n integers or one of `"valid"`,
            `"same"`.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch_size, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch_size, channels, ...)`.
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        groups: A positive integer specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved
            separately with `filters / groups` filters. The output is the
            concatenation of all the `groups` results along the channel axis.
            Input channels and `filters` must both be divisible by `groups`.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied.
        use_bias: Boolean, whether the layer uses a bias.
        use_weight_scaling: Boolean, whether the layer uses running weight scaling.
        gain: Float, weight scaling gain.
        lr_multiplier: Float, weight scaling learning rate multiplier.
        kernel_initializer: An initializer for the convolution kernel.
        bias_initializer: An initializer for the bias vector. If None, the default
            initializer will be used.
        kernel_regularizer: Optional regularizer for the convolution kernel.
        bias_regularizer: Optional regularizer for the bias vector.
        activity_regularizer: Optional regularizer function for the output.
        kernel_constraint: Optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: Optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` the weights of this layer will be marked as
            trainable (and listed in `layer.trainable_weights`).
        name: A string, the name of the layer.
    """

    def __init__(self,
                 padding=0,
                 pad_type='constant',
                 pad_constant_values=0,
                 fir=None,
                 fir_factor=None,
                 fir_stride=1,
                 fir_normalize=True,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier

        super().__init__(
            padding=get_str_padding(padding),
            kernel_initializer=get_initializer(kernel_initializer,
                                               use_weight_scaling,
                                               lr_multiplier),
            **kwargs)

        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        self.pad = get_padding_layer(
            rank=self.rank,
            padding=padding,
            pad_type=pad_type,
            constant_values=pad_constant_values,
            data_format=self.data_format)
        self.fir = get_filter_layer(
            filter=fir,
            factor=fir_factor or self._fir_factor_from_stride(self.strides),
            stride=fir_stride,
            kernel_normalize=fir_normalize,
            data_format=self.data_format)
        self.noise = get_noise_layer(
            noise=noise,
            strength=noise_strength,
            trainable=noise_trainable)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling()

    def call(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)

        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        outputs = self._convolution_op(inputs, kernel)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(bias, (1, self.filters, 1))
                outputs += bias
            # Handle multiple batch dimensions.
            elif output_rank is not None and output_rank > 2 + self.rank:
                def _apply_fn(o):
                    return tf.nn.bias_add(
                        o, bias, data_format=self._tf_data_format)

                outputs = nn_ops.squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = tf.nn.bias_add(
                    outputs, bias, data_format=self._tf_data_format)

        if self.activation:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        spatial_output_shape = super()._spatial_output_shape(spatial_input_shape)
        if self.pad is None:
            return spatial_output_shape

        # spatial output shape for custom pad
        padding = self.pad.padding
        return [
            length + sum(padding[self._spatial_axes[i]]) // self.strides[i]
            for i, length in enumerate(spatial_output_shape)
        ]

    def _check_weight_scaling(self):
        # kernel.shape: (kernel_size, kernel_size, channels//groups, filters)
        if self.use_weight_scaling:
            fan_in = np.prod(self.kernel.shape[:-1])
            self.runtime_coef = self.gain / np.sqrt(fan_in)
            self.runtime_coef *= self.lr_multiplier

    def _fir_factor_from_stride(self, stride):
        if isinstance(stride, int):
            return stride
        if hasattr(stride, '__len__'):
            set_stride = set(stride)
            if len(set_stride) == 1:
                return set_stride.pop()
        raise ValueError('Input `stride` does not support fir filter.')

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
            'use_weight_scaling': self.use_weight_scaling,
            'gain': self.gain,
            'lr_multiplier': self.lr_multiplier,
            'pad': get_layer_config(self.pad),
            'fir': get_layer_config(self.fir),
            'noise': get_layer_config(self.noise)
        })
        return config


class Conv1D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 activation=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class Conv2D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 activation=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class Conv3D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=(0, 0, 0),
                 activation=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class TransposeConv(Conv):
    """
    Inherited from the official tf implementation.
    (edited by https://github.com/kynk94)

    Abstract N-D transposed convolution layer
    (private, used as implementation base).
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 fir=None,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_bias=False,
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
                 **kwargs):
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            fir=fir,
            noise=noise,
            noise_strength=noise_strength,
            noise_trainable=noise_trainable,
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
            **kwargs)
        if output_padding is None:
            self.output_padding = None
        else:
            self.output_padding = conv_utils.normalize_tuple(
                output_padding, rank, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad < stride:
                    continue
                raise ValueError(f'Stride {self.strides} must be greater '
                                 f'than output padding {self.output_padding}')

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != self.rank + 2:
            raise ValueError(f'Inputs should have rank {self.rank + 2}.'
                             f'Received input shape: {input_shape}')
        if input_shape.dims[self._channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be '
                             'defined. Found `None`.')
        input_channel = int(input_shape[self._channel_axis])
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the '
                f'number of groups. Received groups={self.groups}, but the '
                f'input has {input_channel} channels (full input shape is '
                f'{input_shape}).')
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=self.rank + 2, axes={self._channel_axis: input_channel})
        kernel_shape = self.kernel_size + (self.filters // self.groups,
                                           input_channel)

        self.kernel = self.add_weight(
            'kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        spatial_output_shape = self._spatial_output_shape(
            input_shape[axis] for axis in self._spatial_axes)
        if self.data_format == 'channels_first':
            self.spatial_output_shape = (self.filters, *spatial_output_shape)
        else:
            self.spatial_output_shape = (*spatial_output_shape, self.filters)

        conv_kwargs = {
            'strides': self.strides,
            'padding': self.padding.upper(),
            'dilations': self.dilation_rate,
            'data_format': self._tf_data_format}
        if self.rank == 1:
            conv_op = tf.nn.conv1d_transpose
        elif self.rank == 2:
            if self.dilation_rate[0] == 1:
                conv_op = tf.nn.conv2d_transpose
            else:
                conv_op = tf.keras.backend.conv2d_transpose
                conv_kwargs.update({
                    'dilation_rate': conv_kwargs.pop('dilations'),
                    'data_format': self.data_format})
        else:
            conv_op = tf.nn.conv3d_transpose

        self._convolution_op = functools.partial(
            conv_op,
            **conv_kwargs)
        self._check_weight_scaling()
        self.built = True

    def call(self, inputs):
        outputs = inputs

        if self.pad:
            outputs = self.pad(outputs)

        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        outputs = self._convolution_op(
            outputs,
            kernel,
            tf.stack((tf.shape(outputs)[0], *self.spatial_output_shape)))

        if not context.executing_eagerly():
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.fir:
            outputs = self.fir(outputs)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            outputs = tf.nn.bias_add(
                outputs, bias, data_format=self._tf_data_format)

        if self.activation:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        if self.output_padding is None:
            output_padding = (None,) * self.rank
        else:
            output_padding = self.output_padding
        spatial_output_shape = [
            conv_utils.deconv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                output_padding=output_padding[i],
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]
        if self.pad is None:
            return spatial_output_shape

        # spatial output shape for custom pad
        padding = self.pad.padding
        return [
            length + sum(padding[self._spatial_axes[i]]) * self.strides[i]
            for i, length in enumerate(spatial_output_shape)
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_padding': self.output_padding
        })
        return config


class TransposeConv1D(TransposeConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 activation=None,
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class TransposeConv2D(TransposeConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 activation=None,
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class TransposeConv3D(TransposeConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=(0, 0, 0),
                 activation=None,
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class DecompTransConv(Conv):
    """
    Decomposed Transposed Convolution.

    Decomposed along spatial axes, not channel(depth) axis.
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 fir=None,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_bias=False,
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
                 **kwargs):
        if rank <= 1:
            raise ValueError('`rank` of DecomposeTransposeConv should '
                             'greater than 1.')
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            fir=fir,
            noise=noise,
            noise_strength=noise_strength,
            noise_trainable=noise_trainable,
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
            **kwargs)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != self.rank + 2:
            raise ValueError(f'Inputs should have rank {self.rank + 2}.'
                             f'Received input shape: {input_shape}')
        if input_shape.dims[self._channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be '
                             'defined. Found `None`.')
        input_channel = int(input_shape[self._channel_axis])
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the '
                f'number of groups. Received groups={self.groups}, but the '
                f'input has {input_channel} channels (full input shape is '
                f'{input_shape}).')
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=self.rank + 2, axes={self._channel_axis: input_channel})
        kernel_shapes = []
        filters = self.filters // self.groups
        for i in range(self.rank):
            kernel_size = [1] * self.rank
            kernel_size[i] = self.kernel_size[i]
            if i == 0:
                shape = (*kernel_size, filters, input_channel)
            else:
                shape = (*kernel_size, filters, filters)
            kernel_shapes.append(shape)

        self.kernels = [
            self.add_weight(
                f'kernel_{i}',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)
            for i, kernel_shape in enumerate(kernel_shapes)
        ]
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        temporal_output_shape = []
        for axis in self._spatial_axes:
            shape = conv_utils.deconv_output_length(
                input_shape[axis],
                1,
                padding=self.padding,
                output_padding=None,
                stride=1,
                dilation=1)
            if self.pad is not None:
                shape += sum(self.pad.padding[axis])
            temporal_output_shape.append(shape)

        self.spatial_output_shapes = []
        spatial_output_shape = self._spatial_output_shape(
            input_shape[axis] for axis in self._spatial_axes)
        for i in range(self.rank):
            output_shape = spatial_output_shape[:i+1] + \
                temporal_output_shape[i+1:]
            if self.data_format == 'channels_first':
                output_shape = (self.filters, *output_shape)
            else:
                output_shape = (*output_shape, self.filters)
            self.spatial_output_shapes.append(output_shape)

        conv_kwargs = {
            'padding': self.padding.upper(),
            'dilations': self.dilation_rate,
            'data_format': self._tf_data_format}
        if self.rank == 1:
            conv_op = tf.nn.conv1d_transpose
        elif self.rank == 2:
            if self.dilation_rate[0] == 1:
                conv_op = tf.nn.conv2d_transpose
            else:
                conv_op = tf.keras.backend.conv2d_transpose
                conv_kwargs.update({
                    'dilation_rate': conv_kwargs.pop('dilations'),
                    'data_format': self.data_format})
        else:
            conv_op = tf.nn.conv3d_transpose

        self._convolution_ops = []
        for i in range(self.rank):
            strides = [1] * self.rank
            strides[i] = self.strides[i]
            self._convolution_ops.append(
                functools.partial(
                    conv_op,
                    strides=tuple(strides),
                    **conv_kwargs)
            )

        if self.use_weight_scaling:
            self.runtime_coefs = []
            for kernel_shape in kernel_shapes:
                fan_in = np.prod(kernel_shape[:-1])
                runtime_coef = self.gain / np.sqrt(fan_in)
                self.runtime_coefs.append(runtime_coef * self.lr_multiplier)
        self.built = True

    def call(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)

        outputs = inputs
        for i in range(self.rank):
            if self.use_weight_scaling:
                kernel = self.kernels[i] * self.runtime_coefs[i]
            else:
                kernel = self.kernels[i]
            outputs = self._convolution_ops[i](
                outputs, kernel, tf.stack((tf.shape(inputs)[0],
                                           *self.spatial_output_shapes[i])))

        if not context.executing_eagerly():
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.fir:
            outputs = self.fir(outputs)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            outputs = tf.nn.bias_add(
                outputs, bias, data_format=self._tf_data_format)

        if self.activation:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        spatial_output_shape = [
            conv_utils.deconv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                output_padding=None,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]
        if self.pad is None:
            return spatial_output_shape

        # spatial output shape for custom pad
        padding = self.pad.padding
        return [
            length + sum(padding[self._spatial_axes[i]]) * self.strides[i]
            for i, length in enumerate(spatial_output_shape)
        ]


class DecompTransConv2D(DecompTransConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class DecompTransConv3D(DecompTransConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=(0, 0, 0),
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class DownConv(Conv):
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 factor=None,
                 size=None,
                 method='nearest',
                 preserve_aspect_ratio=False,
                 antialias=False,
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_bias=False,
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
                 **kwargs):
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            noise=noise,
            noise_strength=noise_strength,
            noise_trainable=noise_trainable,
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
            **kwargs)
        self.downsample = Downsample(
            factor=factor,
            size=size,
            method=method,
            preserve_aspect_ratio=preserve_aspect_ratio,
            antialias=antialias,
            data_format=data_format)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        outputs = self.downsample(inputs)
        return super().call(outputs)

    def _spatial_output_shape(self, spatial_input_shape):
        if hasattr(self.downsample.factor, '__len__'):
            factor = self.downsample.factor
        else:
            factor = (self.downsample.factor,) * self.rank
        return super()._spatial_output_shape(
            length // factor[i]
            for i, length in enumerate(spatial_input_shape))

    def get_config(self):
        config = super().get_config()
        config.update({
            'downsample': get_layer_config(self.downsample)
        })
        return config


class DownConv1D(DownConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 factor=2,
                 method='nearest',
                 activation=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            factor=factor,
            method=method,
            activation=activation,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class DownConv2D(DownConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 factor=2,
                 method='nearest',
                 activation=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            factor=factor,
            method=method,
            activation=activation,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class DownConv3D(DownConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=(0, 0, 0),
                 factor=2,
                 method='nearest',
                 activation=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            factor=factor,
            method=method,
            activation=activation,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class UpConv(Conv):
    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 factor=None,
                 size=None,
                 method='nearest',
                 preserve_aspect_ratio=False,
                 antialias=False,
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 fir=None,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_bias=False,
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
                 **kwargs):
        super().__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            fir=fir,
            fir_factor=factor,
            noise=noise,
            noise_strength=noise_strength,
            noise_trainable=noise_trainable,
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
            **kwargs)
        self.upsample = Upsample(
            factor=factor,
            size=size,
            method=method,
            preserve_aspect_ratio=preserve_aspect_ratio,
            antialias=antialias,
            data_format=data_format)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        outputs = self.upsample(inputs)
        if self.fir:
            outputs = self.fir(outputs)
        return super().call(outputs)

    def _spatial_output_shape(self, spatial_input_shape):
        if hasattr(self.upsample.factor, '__len__'):
            factor = self.upsample.factor
        else:
            factor = (self.upsample.factor,) * self.rank
        return super()._spatial_output_shape(
            int(length * factor[i])
            for i, length in enumerate(spatial_input_shape))

    def get_config(self):
        config = super().get_config()
        config.update({
            'upsample': get_layer_config(self.upsample)
        })
        return config


class UpConv1D(UpConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding=0,
                 factor=2,
                 method='nearest',
                 activation=None,
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            factor=factor,
            method=method,
            activation=activation,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class UpConv2D(UpConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 factor=2,
                 method='nearest',
                 activation=None,
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            factor=factor,
            method=method,
            activation=activation,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class UpConv3D(UpConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding=(0, 0, 0),
                 factor=2,
                 method='nearest',
                 activation=None,
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            factor=factor,
            method=method,
            activation=activation,
            fir=fir,
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            **kwargs)


class SubPixelConv2D(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding=(0, 0),
                 factor=2,
                 use_icnr_initializer=False,
                 activation=None,
                 fir=None,
                 noise=None,
                 use_bias=False,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        self.factor = factor
        self.use_icnr_initializer = use_icnr_initializer
        if use_weight_scaling:
            stddev = 1.0 / lr_multiplier
            kernel_initializer = tf.initializers.random_normal(0, stddev)
        if use_icnr_initializer:
            kernel_initializer = ICNR(self.factor, kernel_initializer)

        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            fir=fir,
            fir_factor=factor // self._fir_factor_from_stride(strides),
            noise=noise,
            use_bias=use_bias,
            use_weight_scaling=False,  # `use_weight_scaling` should be False.
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            **kwargs)

        # reinitialize `use_weight_scaling` after super().__init__()
        self.use_weight_scaling = use_weight_scaling

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the '
                f'number of groups. Received groups={self.groups}, but the '
                f'input has {input_channel} channels (full input shape is '
                f'{input_shape}).')
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters * self.factor**2)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding

        self._convolution_op = functools.partial(
            tf.nn.convolution,
            strides=list(self.strides),
            padding=tf_padding,
            dilations=list(self.dilation_rate),
            data_format=self._tf_data_format,
            name=self.__class__.__name__)
        self._check_weight_scaling()
        self.built = True


    def call(self, inputs):
        if self.pad:
            inputs = self.pad(inputs)

        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        outputs = self._convolution_op(inputs, kernel)
        outputs = tf.nn.depth_to_space(input=outputs,
                                       block_size=self.factor,
                                       data_format=self._tf_data_format)

        if self.fir:
            outputs = self.fir(outputs)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = tf.reshape(bias, (1, self.filters, 1))
                outputs += bias
            # Handle multiple batch dimensions.
            elif output_rank is not None and output_rank > 2 + self.rank:
                def _apply_fn(o):
                    return tf.nn.bias_add(
                        o, bias, data_format=self._tf_data_format)

                outputs = nn_ops.squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = tf.nn.bias_add(
                    outputs, bias, data_format=self._tf_data_format)

        if self.activation:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            length * self.factor
            for length in super()._spatial_output_shape(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tf.TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters])
        return tf.TensorShape(
            input_shape[:batch_rank] + [self.filters] +
            self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def get_config(self):
        config = super().get_config()
        config.update({
            'factor': self.factor,
            'use_icnr_initializer': self.use_icnr_initializer
        })
        return config
