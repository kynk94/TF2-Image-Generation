import numpy as np
import tensorflow as tf
from .normalizations import Normalization
from .utils import get_activation_layer, get_noise_layer
from .utils import get_layer_config


class Linear(tf.keras.layers.Dense):
    """
    Inherited from the official tf implementation Dense.
    (edited by https://github.com/kynk94)

    Just your regular linearly-connected NN layer.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='glorot_uniform',
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
        self.noise = get_noise_layer(noise=noise,
                                     strength=noise_strength,
                                     trainable=noise_trainable)
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

        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
        # We use embedding_lookup_sparse as a more efficient matmul operation for
        # large sparse input tensors. The op will result in a sparse gradient, as
        # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
        # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, tf.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = tf.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = tf.nn.embedding_lookup_sparse(
                    kernel, ids, weights, combiner='sum')
            else:
                outputs = tf.raw_ops.MatMul(a=inputs, b=kernel)
        # Broadcast kernel to inputs.
        else:
            outputs = tf.tensordot(inputs, kernel, [[rank - 1], [0]])
        # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [kernel.shape[-1]]
                outputs.set_shape(output_shape)

        if self.noise:
            outputs = self.noise(outputs)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
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
        return config


class LinearBlock(tf.keras.Model):
    """
    Linear Block.

    Linear block consists of linear, normalization, and activation layers.
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 noise=None,
                 noise_strength=0.0,
                 noise_trainable=True,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 kernel_initializer='glorot_uniform',
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
        self.normalization = Normalization(
            normalization=normalization,
            momentum=norm_momentum,
            group=norm_group,
            trainable=trainable,
            name='normalization') if normalization else None

        # activation layer
        self.activation = get_activation_layer(activation, activation_alpha)

        # linear layer
        self.linear = Linear(
            units=units,
            activation=None,
            use_bias=use_bias,
            noise=noise,
            noise_strength=noise_strength,
            noise_trainable=noise_trainable,
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
            name='linear')

    def call(self, inputs):
        outputs = inputs
        # normalization -> activation -> linear
        if self.normalization_first:
            if self.normalization:
                outputs = self.normalization(outputs)
            if self.activation:
                outputs = self.activation(outputs)
            outputs = self.linear(outputs)
        # activation -> linear -> normalization
        elif self.activation_first:
            if self.activation:
                outputs = self.activation(outputs)
            outputs = self.linear(outputs)
            if self.normalization:
                outputs = self.normalization(outputs)
        # linear -> normalization -> activation
        else:
            outputs = self.linear(outputs)
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
            'linear': get_layer_config(self.linear),
            'normalization': get_layer_config(self.normalization),
            'activation': get_layer_config(self.activation)
        }
        return config
