"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers as K_layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.utils import conv_utils, generic_utils
from .conv_blocks import ConvBlock, DownConvBlock, UpConvBlock, TransConvBlock
from .utils import get_activation_layer, get_layer_config, kwargs_as_iterable


class ResidualMultiplier(K_layers.Layer):
    def __init__(self,
                 value=1.0,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.init_value = value

    def build(self, input_shape):
        self.multiplier = self.add_weight(
            name='multiplier',
            shape=(),
            initializer=tf.initializers.Constant(self.init_value),
            trainable=True,
            dtype=self.dtype)

    def call(self, inputs):
        return inputs * self.multiplier

    def get_config(self):
        config = super().get_config()
        config.update({
            'init_value': self.value
        })
        return config


class BaseResBlock(Model):
    def __init__(self,
                 rank,
                 depth,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 data_format=None,
                 activation_alpha=0.3,
                 output_activation=None,
                 use_multiplier=False,
                 multiplier_value=1.0,
                 multiplier_trainable=True,
                 conv_op='conv',
                 use_shortcut=False,
                 shortcut_op='conv',
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.depth = depth
        self.use_shortcut = use_shortcut
        self.conv_op = self._get_op(conv_op, is_shortcut=False)
        self.shortcut_op = self._get_op(shortcut_op, is_shortcut=True)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self._channel_axis = self._get_channel_axis()

        self.filters, self.kernel_size, self.strides, self.padding = \
            kwargs_as_iterable(iter_len=depth,
                               filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding).values()

        # output activation layer
        self.output_activation = get_activation_layer(output_activation,
                                                      activation_alpha)

        # residual multiplier
        if not use_multiplier or (multiplier_value == 1.0 and
                                  not multiplier_trainable):
            self.multiplier = None
        else:
            self.multiplier = ResidualMultiplier(
                value=multiplier_value,
                trainable=multiplier_trainable,
                name='residual_multiplier')

        self.conv_blocks = getattr(self, 'conv_blocks', None)
        self.conv_block_args = getattr(self, 'conv_block_args', dict())
        self.shortcut_block_args = getattr(self, 'shortcut_block_args', dict())
        self.other_args = getattr(self, 'other_args', dict())

    def build(self, input_shape):
        self.conv_blocks = self._build_conv_blocks(input_shape)
        self.shortcut = self._build_shortcut_block()
        super().build(input_shape)

    def _build_conv_blocks(self, input_shape):
        channel = input_shape[self._channel_axis]
        if not self.use_shortcut and self.filters[-1] != channel:
            self.filters = (*self.filters[:-1], channel)

        conv_blocks = []
        normalization_first = self.conv_block_args['normalization_first']
        activation_first = self.conv_block_args['activation_first']
        for i in range(self.depth):
            if i == self.depth - 1 and not (normalization_first or
                                            activation_first):
                self.conv_block_args['activation'] = None
            conv_args = self.conv_block_args.copy()
            if i == 0:
                conv_args.update(self.other_args)
            conv_blocks.append(
                self.conv_op[i](
                    rank=self.rank,
                    filters=self.filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.strides[i],
                    padding=self.padding[i],
                    data_format=self.data_format,
                    name=f'{self._get_block_name(self.conv_op[i])}_{i}',
                    **conv_args))
        return conv_blocks

    def _build_shortcut_block(self):
        if not self.use_shortcut:
            return None
        shortcut_args = self.shortcut_block_args.copy()
        shortcut_args.update(self.other_args)
        return self.shortcut_op(
            rank=self.rank,
            filters=self.filters[-1],
            data_format=self.data_format,
            name=f'shortcut_{self._get_block_name(self.shortcut_op)}',
            **shortcut_args)

    def call(self, inputs):
        outputs = inputs
        # convolution block
        for conv_block in self.conv_blocks:
            outputs = conv_block(outputs)
        # residual multiplier
        if self.multiplier:
            outputs = self.multiplier(outputs)
        # shortcut block
        if self.shortcut:
            inputs = self.shortcut(inputs)
        # residual add
        outputs += inputs

        if self.output_activation:
            outputs = self.output_activation(outputs)
        return outputs

    def _get_op(self, op, is_shortcut=False):
        if op is None:
            return None
        l_op = op.lower()
        if l_op == 'conv':
            op = ConvBlock
        elif l_op in {'downsample_conv', 'down_conv', 'down'}:
            op = DownConvBlock
        elif l_op in {'upsample_conv', 'up_conv', 'up'}:
            op = UpConvBlock
        elif l_op in {'transpose_conv', 'trans_conv', 'trans'}:
            op = TransConvBlock
        else:
            raise ValueError(f'Operation {op} not supported.')
        if is_shortcut:
            return op
        return (op,) + (ConvBlock,) * (self.depth - 1)

    def _get_block_name(self, block):
        block_name = generic_utils.to_snake_case(block.__name__).split('_')
        return '_'.join(block_name[:-1]) + f'_{self.rank}d_block'

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        return self.rank + 1

    def get_config(self):
        config = {
            'name': self.name,
            'depth': self.depth,
            'shortcut': get_layer_config(self.shortcut),
            'output_activation': get_layer_config(self.output_activation)
        }
        config.update({
            f'conv_block_{i}': get_layer_config(self.conv_blocks[i])
            for i in range(self.depth)
        })
        return config


class ResBlock(BaseResBlock):
    def __init__(self,
                 rank,
                 depth,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 shortcut_kernel_size=1,
                 shortcut_strides=1,
                 shortcut_padding=0,
                 dilation_rate=1,
                 groups=1,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 output_activation=None,
                 use_multiplier=False,
                 multiplier_value=1.0,
                 multiplier_trainable=True,
                 use_shortcut=False,
                 use_shortcut_bias=False,
                 shortcut_normalization=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=rank,
            depth=depth,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation_alpha=activation_alpha,
            output_activation=output_activation,
            use_multiplier=use_multiplier,
            multiplier_value=multiplier_value,
            multiplier_trainable=multiplier_trainable,
            conv_op='conv',
            use_shortcut=use_shortcut,
            shortcut_op='conv',
            trainable=trainable,
            name=name,
            **kwargs)

        common_conv_args = {
            'pad_type': pad_type,
            'pad_constant_values': pad_constant_values,
            'dilation_rate': dilation_rate,
            'groups': groups,
            'norm_momentum': norm_momentum,
            'norm_group': norm_group,
            'use_weight_scaling': use_weight_scaling,
            'gain': gain,
            'lr_multiplier': lr_multiplier,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint}

        # convolution blocks
        self.conv_block_args = {
            'noise': noise,
            'noise_strength': noise_strength,
            'noise_trainable': noise_trainable,
            'normalization': normalization,
            'normalization_first': normalization_first,
            'activation': activation,
            'activation_alpha': activation_alpha,
            'activation_first': activation_first,
            'use_bias': use_bias,
            'use_spectral_norm': use_spectral_norm,
            'spectral_iteration': spectral_iteration,
            **common_conv_args}

        # shortcut block
        self.shortcut_block_args = {
            'kernel_size': shortcut_kernel_size,
            'strides': shortcut_strides,
            'padding': shortcut_padding,
            'normalization': shortcut_normalization,
            'normalization_first': normalization_first,
            'activation': None,
            'use_bias': use_shortcut_bias,
            **common_conv_args}


class ResBlock2D(ResBlock):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 shortcut_kernel_size=1,
                 shortcut_strides=1,
                 shortcut_padding=0,
                 depth=2,
                 use_bias=False,
                 pad_type='zero',
                 normalization=None,
                 normalization_first=False,
                 activation=None,
                 activation_first=False,
                 output_activation=None,
                 use_shortcut=False,
                 use_shortcut_bias=False,
                 shortcut_normalization=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            shortcut_kernel_size=shortcut_kernel_size,
            shortcut_strides=shortcut_strides,
            shortcut_padding=shortcut_padding,
            depth=depth,
            use_bias=use_bias,
            pad_type=pad_type,
            normalization=normalization,
            normalization_first=normalization_first,
            activation=activation,
            activation_first=activation_first,
            output_activation=output_activation,
            use_shortcut=use_shortcut,
            use_shortcut_bias=use_shortcut_bias,
            shortcut_normalization=shortcut_normalization,
            **kwargs)


class DownResBlock(BaseResBlock):
    def __init__(self,
                 rank,
                 depth,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 shortcut_kernel_size=1,
                 shortcut_strides=1,
                 shortcut_padding=0,
                 factor=None,
                 size=None,
                 method='nearest',
                 dilation_rate=1,
                 groups=1,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 output_activation=None,
                 use_multiplier=False,
                 multiplier_value=1.0,
                 multiplier_trainable=True,
                 use_shortcut=True,
                 use_shortcut_bias=False,
                 shortcut_normalization=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=rank,
            depth=depth,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation_alpha=activation_alpha,
            output_activation=output_activation,
            use_multiplier=use_multiplier,
            multiplier_value=multiplier_value,
            multiplier_trainable=multiplier_trainable,
            conv_op='down',
            use_shortcut=use_shortcut,
            shortcut_op='down',
            trainable=trainable,
            name=name,
            **kwargs)

        common_conv_args = {
            'pad_type': pad_type,
            'pad_constant_values': pad_constant_values,
            'dilation_rate': dilation_rate,
            'groups': groups,
            'norm_momentum': norm_momentum,
            'norm_group': norm_group,
            'use_weight_scaling': use_weight_scaling,
            'gain': gain,
            'lr_multiplier': lr_multiplier,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint}

        self.other_args = {
            'factor': factor,
            'size': size,
            'method': method,
        }

        # convolution blocks
        self.conv_block_args = {
            'noise': noise,
            'noise_strength': noise_strength,
            'noise_trainable': noise_trainable,
            'normalization': normalization,
            'normalization_first': normalization_first,
            'activation': activation,
            'activation_alpha': activation_alpha,
            'activation_first': activation_first,
            'use_bias': use_bias,
            'use_spectral_norm': use_spectral_norm,
            'spectral_iteration': spectral_iteration,
            **common_conv_args}

        # shortcut block
        self.shortcut_block_args = {
            'kernel_size': shortcut_kernel_size,
            'strides': shortcut_strides,
            'padding': shortcut_padding,
            'normalization': shortcut_normalization,
            'normalization_first': normalization_first,
            'activation': None,
            'use_bias': use_shortcut_bias,
            **common_conv_args}


class DownResBlock2D(DownResBlock):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 shortcut_kernel_size=1,
                 shortcut_strides=1,
                 shortcut_padding=0,
                 depth=2,
                 factor=2,
                 method='nearest',
                 use_bias=False,
                 pad_type='zero',
                 normalization=None,
                 normalization_first=False,
                 activation=None,
                 activation_first=False,
                 output_activation=None,
                 use_shortcut=True,
                 use_shortcut_bias=False,
                 shortcut_normalization=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            shortcut_kernel_size=shortcut_kernel_size,
            shortcut_strides=shortcut_strides,
            shortcut_padding=shortcut_padding,
            depth=depth,
            factor=factor,
            method=method,
            use_bias=use_bias,
            pad_type=pad_type,
            normalization=normalization,
            normalization_first=normalization_first,
            activation=activation,
            activation_first=activation_first,
            output_activation=output_activation,
            use_shortcut=use_shortcut,
            use_shortcut_bias=use_shortcut_bias,
            shortcut_normalization=shortcut_normalization,
            trainable=trainable,
            name=name,
            **kwargs)


class UpResBlock(BaseResBlock):
    def __init__(self,
                 rank,
                 depth,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 shortcut_kernel_size=1,
                 shortcut_strides=1,
                 shortcut_padding=0,
                 factor=None,
                 size=None,
                 method='nearest',
                 dilation_rate=1,
                 groups=1,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_bias=False,
                 use_spectral_norm=False,
                 spectral_iteration=1,
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
                 data_format=None,
                 pad_type='constant',
                 pad_constant_values=0,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
                 output_activation=None,
                 use_multiplier=False,
                 multiplier_value=1.0,
                 multiplier_trainable=True,
                 use_shortcut=True,
                 use_shortcut_bias=False,
                 shortcut_normalization=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=rank,
            depth=depth,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation_alpha=activation_alpha,
            output_activation=output_activation,
            use_multiplier=use_multiplier,
            multiplier_value=multiplier_value,
            multiplier_trainable=multiplier_trainable,
            conv_op='up',
            use_shortcut=use_shortcut,
            shortcut_op='up',
            trainable=trainable,
            name=name,
            **kwargs)

        common_conv_args = {
            'pad_type': pad_type,
            'pad_constant_values': pad_constant_values,
            'dilation_rate': dilation_rate,
            'groups': groups,
            'norm_momentum': norm_momentum,
            'norm_group': norm_group,
            'use_weight_scaling': use_weight_scaling,
            'gain': gain,
            'lr_multiplier': lr_multiplier,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint}

        self.other_args = {
            'factor': factor,
            'size': size,
            'method': method,
        }

        # convolution blocks
        self.conv_block_args = {
            'noise': noise,
            'noise_strength': noise_strength,
            'noise_trainable': noise_trainable,
            'normalization': normalization,
            'normalization_first': normalization_first,
            'activation': activation,
            'activation_alpha': activation_alpha,
            'activation_first': activation_first,
            'use_bias': use_bias,
            'use_spectral_norm': use_spectral_norm,
            'spectral_iteration': spectral_iteration,
            **common_conv_args}

        # shortcut block
        self.shortcut_block_args = {
            'kernel_size': shortcut_kernel_size,
            'strides': shortcut_strides,
            'padding': shortcut_padding,
            'normalization': shortcut_normalization,
            'normalization_first': normalization_first,
            'activation': None,
            'use_bias': use_shortcut_bias,
            **common_conv_args}


class UpResBlock2D(UpResBlock):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 shortcut_kernel_size=1,
                 shortcut_strides=1,
                 shortcut_padding=0,
                 depth=2,
                 factor=2,
                 method='nearest',
                 use_bias=False,
                 pad_type='zero',
                 normalization=None,
                 normalization_first=False,
                 activation=None,
                 activation_first=False,
                 output_activation=None,
                 use_shortcut=True,
                 use_shortcut_bias=False,
                 shortcut_normalization=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            shortcut_kernel_size=shortcut_kernel_size,
            shortcut_strides=shortcut_strides,
            shortcut_padding=shortcut_padding,
            depth=depth,
            factor=factor,
            method=method,
            use_bias=use_bias,
            pad_type=pad_type,
            normalization=normalization,
            normalization_first=normalization_first,
            activation=activation,
            activation_first=activation_first,
            output_activation=output_activation,
            use_shortcut=use_shortcut,
            use_shortcut_bias=use_shortcut_bias,
            shortcut_normalization=shortcut_normalization,
            trainable=trainable,
            name=name,
            **kwargs)


class ResIdentityBlock2D(ResBlock):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 pad_type='zero',
                 normalization='bn',
                 activation='relu',
                 use_bias=False,
                 use_weight_scaling=False,
                 kernel_initializer='he_normal',
                 output_activation='relu',
                 **kwargs):
        kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        padding = conv_utils.normalize_tuple(padding, 2, 'padding')
        super().__init__(
            rank=2,
            depth=3,
            filters=filters,
            kernel_size=((1, 1), kernel_size, (1, 1)),
            strides=((1, 1), strides, (1, 1)),
            padding=((0, 0), padding, (0, 0)),
            pad_type=pad_type,
            normalization=normalization,
            activation=activation,
            use_bias=use_bias,
            use_weight_scaling=use_weight_scaling,
            kernel_initializer=kernel_initializer,
            output_activation=output_activation,
            **kwargs)
