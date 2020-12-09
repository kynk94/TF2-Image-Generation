"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
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
                    return tf.nn.bias_add(
                        o, bias, data_format=self._tf_data_format)

                outputs = nn_ops.squeeze_batch_dims(
                    outputs, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs = tf.nn.bias_add(
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
            activation=activations.get(activation),
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
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
            activation=activations.get(activation),
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
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
            activation=activations.get(activation),
            use_noise=use_noise,
            noise_strength=noise_strength,
            noise_strength_trainable=noise_strength_trainable,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)


class Conv1DTranspose(ConvBase, convolutional.Conv1DTranspose):
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
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self._get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self._set_noise(use_noise,
                        noise_strength,
                        noise_strength_trainable)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling()

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            t_axis = 2
        else:
            t_axis = 1

        length = inputs_shape[t_axis]
        if self.output_padding is None:
            output_padding = None
        else:
            output_padding = self.output_padding[0]

        # Infer the dynamic output shape:
        out_length = conv_utils.deconv_output_length(
            length, self.kernel_size[0], padding=self.padding,
            output_padding=output_padding, stride=self.strides[0],
            dilation=self.dilation_rate[0])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_length)
        else:
            output_shape = (batch_size, out_length, self.filters)
        data_format = conv_utils.convert_data_format(self.data_format, ndim=3)

        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        output_shape_tensor = tf.stack(output_shape)
        outputs = nn_ops.conv1d_transpose(
            inputs,
            kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=data_format,
            dilations=self.dilation_rate)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            outputs = tf.nn.bias_add(
                outputs,
                bias,
                data_format=data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config


class Conv2DTranspose(ConvBase, convolutional.Conv2DTranspose):
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
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self._get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self._set_noise(use_noise,
                        noise_strength,
                        noise_strength_trainable)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling()

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = None, None
        if inputs.shape.rank is not None:
            dims = inputs.shape.as_list()
            height = dims[h_axis]
            width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(
            height,
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(
            width,
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.keras.backend.conv2d_transpose(
            inputs,
            kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            outputs = tf.nn.bias_add(
                outputs,
                bias,
                data_format=conv_utils.convert_data_format(
                    self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config


class Conv3DTranspose(ConvBase, convolutional.Conv3DTranspose):
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
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=self._get_initializer(
                use_weight_scaling,
                gain,
                lr_multiplier) or kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self._set_noise(use_noise,
                        noise_strength,
                        noise_strength_trainable)

    def build(self, input_shape):
        super().build(input_shape)
        self._check_weight_scaling()

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            d_axis, h_axis, w_axis = 2, 3, 4
        else:
            d_axis, h_axis, w_axis = 1, 2, 3

        depth = inputs_shape[d_axis]
        height = inputs_shape[h_axis]
        width = inputs_shape[w_axis]

        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_d = out_pad_h = out_pad_w = None
        else:
            out_pad_d, out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_depth = conv_utils.deconv_output_length(depth,
                                                    kernel_d,
                                                    padding=self.padding,
                                                    output_padding=out_pad_d,
                                                    stride=stride_d)
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h)
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_depth, out_height,
                            out_width)
            strides = (1, 1, stride_d, stride_h, stride_w)
        else:
            output_shape = (batch_size, out_depth, out_height, out_width,
                            self.filters)
            strides = (1, stride_d, stride_h, stride_w, 1)

        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.nn.conv3d_transpose(
            inputs,
            kernel,
            output_shape_tensor,
            strides,
            data_format=conv_utils.convert_data_format(
                self.data_format, ndim=5),
            padding=self.padding.upper())

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.noise:
            outputs = self.noise(outputs)

        if self.use_bias:
            if self.use_weight_scaling:
                bias = self.bias * self.lr_multiplier
            else:
                bias = self.bias
            outputs = tf.nn.bias_add(
                outputs,
                bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        self._update_config(config)
        return config


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
