import numpy as np
import tensorflow as tf


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
                 kernel_initializer='he_normal',
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
            input_channel = self._get_input_channel(input_shape)
            fan_in = input_channel * np.prod(self.kernel_size)
            self.runtime_coef = self.gain / np.sqrt(fan_in)
            self.runtime_coef *= self.lr_multiplier
            self.kernel = self.kernel * self.runtime_coef

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
