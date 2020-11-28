"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import normalize_data_format
from .conv_blocks import ConvBlock
from .utils import get_activation_layer, get_layer_config, kwargs_as_iterable


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
                 use_shortcut=False,
                 output_activation=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.depth = depth
        self.data_format = normalize_data_format(data_format)

        self.output_activation = get_activation_layer(output_activation,
                                                      activation_alpha)

        self.filters, self.kernel_size, self.strides, self.padding = \
            kwargs_as_iterable(depth=depth,
                               filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding).values()

        self.conv_blocks = []
        for i in range(depth):
            if not normalization_first and i == depth - 1:
                activation = None
            self.conv_blocks.append(
                ConvBlock(rank=rank,
                          filters=self.filters[i],
                          kernel_size=kernel_size[i],
                          strides=self.strides[i],
                          padding=self.padding[i],
                          pad_type=pad_type,
                          pad_constant_values=pad_constant_values,
                          conv_padding=conv_padding,
                          output_padding=output_padding,
                          data_format=self.data_format,
                          dilation_rate=dilation_rate,
                          groups=groups,
                          normalization=normalization,
                          normalization_first=normalization_first,
                          norm_momentum=norm_momentum,
                          norm_group=norm_group,
                          activation=activation,
                          activation_alpha=activation_alpha,
                          activation_first=activation_first,
                          use_bias=use_bias,
                          use_spectral_norm=use_spectral_norm,
                          spectral_iteration=spectral_iteration,
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
                          mode=mode))

        if not use_shortcut:
            self.shortcut = None
        elif mode.lower() in {'downsample', 'down'}:
            self.shortcut = ConvBlock(rank=rank,
                                      filters=self.filters[-1],
                                      kernel_size=1,
                                      strides=1,
                                      use_bias=False,
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
                                      mode=mode)
        elif mode.lower() in {'upsample', 'up'}:
            self.shortcut = ConvBlock(rank=rank,
                                      filters=self.filters[-1],
                                      kernel_size=1,
                                      strides=1,
                                      use_bias=False,
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
                                      mode=mode)

    def call(self, inputs):
        outputs = inputs
        # convolution layer
        for conv_block in self.conv_blocks:
            outputs = conv_block(outputs)
        # shortcut layer
        if self.shortcut:
            inputs = self.shortcut(inputs)
        # residual add
        outputs += inputs

        if self.output_activation:
            outputs = self.output_activation(outputs)
        return outputs

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
