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
from .noise import GaussianNoise
from .utils import get_layer_config


class ConvBase:
    """
    Baseline of Convolution layer.

    Should inherit as first position in custom conv layer.
    """

    def _get_initializer(self, use_weight_scaling, gain, lr_multiplier):
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier
        if use_weight_scaling:
            stddev = 1.0 / lr_multiplier
            return tf.initializers.random_normal(0, stddev)
        return None

    def _check_weight_scaling(self):
        # kernel.shape: (kernel_size, kernel_size, channels//groups, filters)
        if self.use_weight_scaling:
            fan_in = np.prod(self.kernel.shape[:-1])
            self.runtime_coef = self.gain / np.sqrt(fan_in)
            self.runtime_coef *= self.lr_multiplier

    def _set_noise(self,
                   use_noise=False,
                   noise_strength=0.0,
                   noise_strength_trainable=True):
        if not use_noise:
            self.noise = None
        else:
            self.noise = GaussianNoise(
                stddev=1.0,
                strength=noise_strength,
                strength_trainable=noise_strength_trainable,
                channel_same=True)

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        return self.rank + 1

    def _get_spatial_axis(self):
        channel_axis = self._get_channel_axis()
        spatial_axis = list(range(self.rank + 2))
        del spatial_axis[channel_axis]
        del spatial_axis[0]
        return spatial_axis

    def _update_config(self, config):
        config.update({
            'use_weight_scaling':
                self.use_weight_scaling,
            'gain':
                self.gain,
            'lr_multiplier':
                self.lr_multiplier,
            'noise':
                get_layer_config(self.noise)
        })


class Conv(ConvBase, convolutional.Conv):
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
        padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
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
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 **kwargs):
        super().__init__(
            kernel_initializer=self._get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            **kwargs)
        self._channel_axis = self._get_channel_axis()
        self._spatial_axis = self._get_spatial_axis()
        self._set_noise(use_noise,
                        noise_strength,
                        noise_strength_trainable)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling()

    def call(self, inputs):
        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = tf.pad(
                inputs, self._compute_causal_padding(inputs))

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
                    return nn_ops.bias_add(
                        o, bias, data_format=self._tf_data_format)

                outputs = nn_ops.squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = nn_ops.bias_add(
                    outputs, bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config


class Conv1D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
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
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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


class Conv2D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
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
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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


class Conv3D(Conv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 groups=1,
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
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
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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


class ConvTranspose(Conv):
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
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=True,
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
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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
        input_dim = int(input_shape[self._channel_axis])
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=self.rank + 2, axes={self._channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

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

        spatial_input_shape = [input_shape[axis]
                               for axis in self._spatial_axis]
        spatial_output_shape = self._spatial_output_shape(spatial_input_shape)
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
            conv_op = nn_ops.conv1d_transpose
        elif self.rank == 2:
            if self.dilation_rate[0] == 1:
                conv_op = nn_ops.conv2d_transpose
            else:
                conv_op = tf.keras.backend.conv2d_transpose
                conv_kwargs.update({
                    'dilation_rate': conv_kwargs.pop('dilations'),
                    'data_format': self.data_format})
        else:
            conv_op = nn_ops.conv3d_transpose

        self._convolution_op = functools.partial(
            conv_op,
            **conv_kwargs)
        self._check_weight_scaling()
        self.built = True

    def call(self, inputs):
        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        outputs = self._convolution_op(
            inputs,
            kernel,
            tf.stack((tf.shape(inputs)[0], *self.spatial_output_shape)))

        if not context.executing_eagerly():
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            outputs = nn_ops.bias_add(
                outputs, bias, data_format=self._tf_data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        if self.output_padding is None:
            output_padding = (None,) * self.rank
        else:
            output_padding = self.output_padding
        return [
            conv_utils.deconv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                output_padding=output_padding[i],
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_padding': self.output_padding
        })
        return config


class Conv1DTranspose(ConvTranspose):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=True,
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
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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


class Conv2DTranspose(ConvTranspose):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=True,
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
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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


class Conv3DTranspose(ConvTranspose):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1, 1),
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
                 use_bias=True,
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
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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


class UpsampleConv2D(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 size=None,
                 scale=None,
                 method='bilinear',
                 preserve_aspect_ratio=False,
                 antialias=False,
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
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
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
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
        if (size is not None) ^ (scale is None):  # XOR operation
            raise ValueError('Either `size` or `scale` should not be None.')
        self.size = size
        self.scale = scale
        self.method = method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.antialias = antialias

    def build(self, input_shape):
        self._check_scaled_size(input_shape)
        self._resize_op = functools.partial(
            tf.image.resize,
            size=self.size,
            method=self.method,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            antialias=self.antialias,
            name='resize')
        super.build(input_shape)

    def call(self, inputs):
        outputs = self._resize_op(inputs)
        outputs = super().call(outputs)
        return outputs

    def _check_scaled_size(self, input_shape):
        if self.size is not None:
            self.size = conv_utils.normalize_tuple(self.size, 2, 'size')
            return
        spatial_axis = range(len(input_shape))
        del spatial_axis[self.channel_axis]
        del spatial_axis[0]
        self.size = tuple(input_shape[axis] * self.scale
                          for axis in spatial_axis)

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'method': self.method,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'antialias': self.antialias
        })
        return config


class SubPixelConv2D(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 scale=2,
                 use_icnr_initializer=False,
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_noise=False,
                 noise_strength=0.0,
                 noise_strength_trainable=True,
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
        self.scale = scale
        self.use_icnr_initializer = use_icnr_initializer
        if use_weight_scaling:
            stddev = 1.0 / lr_multiplier
            kernel_initializer = tf.initializers.random_normal(0, stddev)
        if use_icnr_initializer:
            kernel_initializer = ICNR(self.scale, kernel_initializer)

        super().__init__(
            filters=filters * (self.scale**2),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=False,  # `use_weight_scaling` should be False.
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

        # reinitialize `use_weight_scaling` after super().__init__()
        self.use_weight_scaling = use_weight_scaling

    def call(self, inputs):
        outputs = super().call(inputs)
        data_format = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
        outputs = tf.nn.depth_to_space(input=outputs,
                                       block_size=self.scale,
                                       data_format=data_format)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'scale': self.scale,
            'use_icnr_initializer': self.use_icnr_initializer
        })
        return config
