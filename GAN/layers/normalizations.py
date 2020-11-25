import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.utils.conv_utils import normalize_data_format


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
            axis=self._get_spatial_axis(axis),
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
        self._set_spatial_axis(input_shape)
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

    def _set_spatial_axis(self, input_shape):
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        del reduction_axes[0]
        self.axis = reduction_axes
