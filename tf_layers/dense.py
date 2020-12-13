import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.ops import core as core_ops
from .utils import get_activation_layer, get_normalization_layer
from .utils import get_layer_config


class Dense(tf.keras.layers.Dense):
    """
    Inherited from the official tf implementation.
    (edited by https://github.com/kynk94)

    Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then `Dense`
    computes the dot product between the `inputs` and the `kernel` along the
    last axis of the `inputs` and axis 1 of the `kernel` (using `tf.tensordot`).
    For example, if input has dimensions `(batch_size, d0, d1)`,
    then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
    along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)`
    (there are `batch_size * d0` such sub-tensors).
    The output in this case will have shape `(batch_size, d0, units)`.

    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Example:

    >>> # Create a `Sequential` model and add a Dense layer as the first layer.
    >>> model = tf.keras.models.Sequential()
    >>> model.add(tf.keras.Input(shape=(16,)))
    >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
    >>> # Now the model will take as input arrays of shape (None, 16)
    >>> # and output arrays of shape (None, 32).
    >>> # Note that after the first layer, you don't need to specify
    >>> # the size of the input anymore:
    >>> model.add(tf.keras.layers.Dense(32))
    >>> model.output_shape
    (None, 32)

    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        use_weight_scaling: Boolean, whether the layer uses running weight scaling.
        gain: Float, weight scaling gain.
        lr_multiplier: Float, weight scaling learning rate multiplier.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier
        if use_weight_scaling:
            stddev = 1.0 / lr_multiplier
            kernel_initializer = tf.initializers.random_normal(0, stddev)
        super().__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_weight_scaling:
            input_shape = tf.TensorShape(input_shape)
            fan_in = np.prod(input_shape[1:])
            self.runtime_coef = self.gain / np.sqrt(fan_in)
            self.runtime_coef *= self.lr_multiplier

    def call(self, inputs):
        if self.use_weight_scaling:
            kernel = self.kernel * self.runtime_coef
        else:
            kernel = self.kernel
        return core_ops.dense(
            inputs,
            kernel,
            self.bias,
            self.activation,
            dtype=self._compute_dtype_object)

    def get_config(self):
        config = super().get_config()
        config.update({
            'use_weight_scaling':
                self.use_weight_scaling,
            'gain':
                self.gain,
            'lr_multiplier':
                self.lr_multiplier
        })
        return config


class DenseBlock(tf.keras.Model):
    """
    Dense Block.

    Dense block consists of dense, normalization, and activation layers.
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='he_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 normalization=None,
                 normalization_first=False,
                 norm_momentum=0.99,
                 norm_group=32,
                 activation=None,
                 activation_first=False,
                 activation_alpha=0.3,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        if normalization_first and activation_first:
            raise ValueError('Only one of `normalization_first` '
                             'or `activation_first` can be True.')
        self.normalization_first = normalization_first
        self.activation_first = activation_first

        # normalization layer
        self.normalization = get_normalization_layer(-1,
                                                     normalization,
                                                     norm_momentum,
                                                     norm_group)

        # activation layer
        self.activation = get_activation_layer(activation, activation_alpha)

        # dense layer
        self.dense = Dense(
            units=units,
            activation=None,
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
            trainable=trainable,
            name='dense')

    def call(self, inputs):
        outputs = inputs
        # normalization -> activation -> dense
        if self.normalization_first:
            if self.normalization:
                outputs = self.normalization(outputs)
            if self.activation:
                outputs = self.activation(outputs)
            outputs = self.dense(outputs)
        # activation -> dense -> normalization
        elif self.activation_first:
            if self.activation:
                outputs = self.activation(outputs)
            outputs = self.dense(outputs)
            if self.normalization:
                outputs = self.normalization(outputs)
        # dense -> normalization -> activation
        else:
            outputs = self.dense(outputs)
            if self.normalization:
                outputs = self.normalization(outputs)
            if self.activation:
                outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'name': self.name,
            'normalization_first': self.normalization_first,
            'activation_first': self.activation_first,
            'dense': get_layer_config(self.dense),
            'normalization': get_layer_config(self.normalization),
            'activation': get_layer_config(self.activation)
        }
        return config
