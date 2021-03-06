import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import layers as K_layers
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow_addons.layers import GroupNormalization, InstanceNormalization
from .utils import get_layer_config


class Normalization(K_layers.Layer):
    def __init__(self,
                 normalization,
                 momentum=0.99,
                 group=32,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 data_format=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.normalization = normalization
        self.momentum = momentum
        self.group = group
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        self.rank = len(input_shape) - 2
        self._channel_axis = self._get_channel_axis()
        self.normalization = self.get_normalization(
            channel_axis=self._channel_axis,
            normalization=self.normalization,
            momentum=self.momentum,
            group=self.group,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale)
        self.built = True

    def call(self, inputs):
        return self.normalization(inputs)

    def get_normalization(self,
                          channel_axis,
                          normalization,
                          momentum=0.99,
                          group=32,
                          epsilon=1e-5,
                          center=True,
                          scale=True):
        if hasattr(normalization, '__call__'):
            return normalization
        if isinstance(normalization, str):
            l_normalization = normalization.lower()
            if l_normalization in {'batch_normalization',
                                   'batch_norm', 'bn'}:
                return BatchNormalization(
                    axis=channel_axis,
                    momentum=momentum,
                    epsilon=epsilon,
                    center=center,
                    scale=scale,
                    name='batch_normalization')
            if l_normalization in {'sync_batch_normalization',
                                   'sync_batch_norm', 'sync_bn', 'sbn'}:
                return SyncBatchNormalization(
                    axis=channel_axis,
                    momentum=momentum,
                    epsilon=epsilon,
                    center=center,
                    scale=scale,
                    name='sync_batch_normalization')
            if l_normalization in {'layer_normalization',
                                   'layer_norm', 'ln'}:
                return LayerNormalization(
                    axis=channel_axis,
                    epsilon=epsilon,
                    center=center,
                    scale=scale,
                    name='layer_normalization')
            if l_normalization in {'instance_normalization',
                                   'instance_norm', 'in'}:
                return InstanceNormalization(
                    axis=channel_axis,
                    epsilon=epsilon,
                    name='instance_normalization')
            if l_normalization in {'group_normalization',
                                   'group_norm', 'gn'}:
                return GroupNormalization(
                    axis=channel_axis,
                    groups=group,
                    epsilon=epsilon,
                    center=center,
                    scale=scale,
                    name='group_normalization')
            if l_normalization in {'filter_response_normalization',
                                   'filter_response_norm', 'frn'}:
                # FilterResponseNormalization is not official implementation.
                # Official need to input axis as spatial, not channel.
                return FilterResponseNormalization(
                    axis=channel_axis,
                    epsilon=epsilon,
                    name='filter_response_normalization')
        raise ValueError(f'Unsupported `normalization`: {normalization}')

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        return self.rank + 1

    def get_config(self):
        return get_layer_config(self.normalization)


class FilterResponseNormalization(tfa.layers.FilterResponseNormalization):
    """
    Inherited from the official tfa implementation.
    (edited by https://github.com/kynk94)

    Filter response normalization layer.

    Filter Response Normalization (FRN), a normalization
    method that enables models trained with per-channel
    normalization to achieve high accuracy. It performs better than
    all other normalization techniques for small batches and is par
    with Batch Normalization for bigger batch sizes.

    Arguments
        axis: Channel axis. The operating axes are set except channel axis.
        epsilon: Small positive float value added to variance to avoid dividing by zero.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        learned_epsilon: (bool) Whether to add another learnable
        epsilon parameter or not.
        name: Optional name for the layer

    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples(batch) axis)
        when using this layer as the first layer in a model.

    Output shape
        Same shape as input.

    References
        - [Filter Response Normalization Layer: Eliminating Batch Dependence
        in the training of Deep Neural Networks]
        (https://arxiv.org/abs/1911.09737)
    """

    def __init__(self,
                 epsilon=1e-6,
                 axis=-1,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 learned_epsilon=False,
                 learned_epsilon_constraint=None,
                 name=None,
                 **kwargs):
        super().__init__(
            epsilon=epsilon,
            axis=axis,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            learned_epsilon=learned_epsilon,
            learned_epsilon_constraint=learned_epsilon_constraint,
            name=name,
            **kwargs)

    def build(self, input_shape):
        self._set_spatial_axes(input_shape)
        self._check_if_input_shape_is_none(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True

    def _check_axis(self, axis):
        if axis == 0:
            raise ValueError(
                'You are trying to normalize your batch axis.'
                'Use tf.layer.batch_normalization instead')
        self.axis = axis

    def _set_spatial_axes(self, input_shape):
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        del reduction_axes[0]
        self.axis = reduction_axes
