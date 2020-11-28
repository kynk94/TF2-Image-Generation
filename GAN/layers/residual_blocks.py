"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import normalize_data_format, normalize_tuple
from .conv_blocks import ConvBlock
from .utils import get_activation_layer, get_layer_config, kwargs_as_iterable


class ResidualMultiplier(tf.keras.layers.Layer):
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


class ResBlock(tf.keras.Model):
    def __init__(self,
                 rank,
                 depth,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 pad_type='constant',
                 pad_constant_values=0,
                 conv_padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_alpha=0.3,
                 activation_first=False,
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
                 mode='down',
                 use_multiplier=False,
                 multiplier_value=1.0,
                 multiplier_trainable=False,
                 use_shortcut=False,
                 use_shortcut_bias=False,
                 shortcut_kernel_size=1,
                 shortcut_strides=1,
                 shortcut_padding=0,
                 shortcut_normalization=None,
                 output_activation=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.depth = depth
        self.data_format = normalize_data_format(data_format)
        self.channel_axis = self._get_channel_axis()
        self.use_shortcut = use_shortcut

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

        common_conv_args = {
            'pad_type': pad_type,
            'pad_constant_values': pad_constant_values,
            'conv_padding': conv_padding,
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
            'bias_constraint': bias_constraint,
            'mode': mode}

        # convolution blocks
        self.conv_blocks = None
        self.conv_block_args = {
            'output_padding': output_padding,
            'normalization': normalization,
            'normalization_first': normalization_first,
            'activation': activation,
            'activation_alpha': activation_alpha,
            'activation_first': activation_first,
            'use_bias': use_bias,
            'use_spectral_norm': use_spectral_norm,
            'spectral_iteration': spectral_iteration,
            **common_conv_args}
        self.filters, self.kernel_size, self.strides, self.padding = \
            kwargs_as_iterable(iter_len=depth,
                               filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding).values()

        # shortcut block
        self.shortcut_block_args = {
            'kernel_size': shortcut_kernel_size,
            'strides': shortcut_strides,
            'padding': shortcut_padding,
            'normalization': shortcut_normalization,
            'activation': None,
            'use_bias': use_shortcut_bias,
            **common_conv_args}

    def _build_conv_blocks(self, input_shape):
        channel = input_shape[self.channel_axis]
        if not self.use_shortcut and self.filters[-1] != channel:
            filters = list(self.filters)
            filters[-1] = channel
            self.filters = tuple(filters)

        conv_blocks = []
        normalization_first = self.conv_block_args['normalization_first']
        activation_first = self.conv_block_args['activation_first']
        for i in range(self.depth):
            if i == self.depth - 1 and not (normalization_first or
                                            activation_first):
                self.conv_block_args['activation'] = None
            conv_blocks.append(
                ConvBlock(
                    rank=self.rank,
                    filters=self.filters[i],
                    kernel_size=self.kernel_size[i],
                    strides=self.strides[i],
                    padding=self.padding[i],
                    data_format=self.data_format,
                    name=f'conv{self.rank}d_block_{i}',
                    **self.conv_block_args))
        return conv_blocks

    def _build_shortcut_block(self):
        if not self.use_shortcut:
            return None
        return ConvBlock(
            rank=self.rank,
            filters=self.filters[-1],
            data_format=self.data_format,
            name=f'shortcut_conv{self.rank}d_block',
            **self.shortcut_block_args)

    def build(self, input_shape):
        self.conv_blocks = self._build_conv_blocks(input_shape)
        self.shortcut = self._build_shortcut_block()
        super().build(input_shape)

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


class ResIdentityBlock2D(ResBlock):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding=1,
                 pad_type='constant',
                 pad_constant_values=0,
                 conv_padding='valid',
                 data_format=None,
                 normalization='bn',
                 norm_momentum=0.99,
                 norm_group=32,
                 activation='relu',
                 activation_alpha=0.3,
                 use_bias=False,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_normal',
                 output_activation='relu',
                 trainable=True,
                 name=None,
                 **kwargs):
        kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')
        strides = normalize_tuple(strides, 2, 'strides')
        padding = normalize_tuple(padding, 2, 'padding')
        super().__init__(
            rank=2,
            depth=3,
            filters=filters,
            kernel_size=((1, 1), kernel_size, (1, 1)),
            strides=((1, 1), strides, (1, 1)),
            padding=((0, 0), padding, (0, 0)),
            pad_type=pad_type,
            pad_constant_values=pad_constant_values,
            conv_padding=conv_padding,
            output_padding=None,
            data_format=data_format,
            dilation_rate=(1, 1),
            groups=1,
            normalization=normalization,
            normalization_first=False,
            norm_momentum=norm_momentum,
            norm_group=norm_group,
            activation=activation,
            activation_alpha=activation_alpha,
            activation_first=False,
            use_bias=use_bias,
            use_spectral_norm=False,
            spectral_iteration=1,
            use_weight_scaling=use_weight_scaling,
            gain=gain,
            lr_multiplier=lr_multiplier,
            kernel_initializer=kernel_initializer,
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            mode='down',
            use_multiplier=False,
            multiplier_value=1.0,
            multiplier_trainable=False,
            use_shortcut=False,
            use_shortcut_bias=False,
            shortcut_kernel_size=1,
            shortcut_strides=1,
            shortcut_padding=0,
            shortcut_normalization=None,
            output_activation=output_activation,
            trainable=trainable,
            name=name,
            **kwargs)
