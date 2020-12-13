import numpy as np
import tensorflow as tf
from tensorflow.python.distribute import sharded_variable


class Embedding(tf.keras.layers.Embedding):
    """
    Inherited from the official tf implementation.
    (edited by https://github.com/kynk94)

    Turns positive integers (indexes) into dense vectors of fixed size.

    e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

    This layer can only be used as the first layer in a model.

    Example:

    >>> model = tf.keras.Sequential()
    >>> model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
    >>> # The model will take as input an integer matrix of size (batch,
    >>> # input_length), and the largest integer (i.e. word index) in the input
    >>> # should be no larger than 999 (vocabulary size).
    >>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
    >>> # dimension.
    >>> input_array = np.random.randint(1000, size=(32, 10))
    >>> model.compile('rmsprop', 'mse')
    >>> output_array = model.predict(input_array)
    >>> print(output_array.shape)
    (32, 10, 64)

    Arguments:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        use_weight_scaling: Boolean, whether the layer uses running weight scaling.
        gain: Float, weight scaling gain.
        lr_multiplier: Float, weight scaling learning rate multiplier.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special "padding"
            value that should be masked out.
            This is useful when using recurrent layers
            which may take variable length input.
            If this is `True`, then all subsequent layers
            in the model need to support masking or an exception will be raised.
            If mask_zero is set to True, as a consequence, index 0 cannot be
            used in the vocabulary (input_dim should equal size of
            vocabulary + 1).
        input_length: Length of input sequences, when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).

    Input shape:
        2D tensor with shape: `(batch_size, input_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, input_length, output_dim)`.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 embeddings_initializer='he_normal',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier
        if use_weight_scaling:
            stddev = 1.0 / lr_multiplier
            embeddings_initializer = tf.initializers.random_normal(0, stddev)
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_weight_scaling:
            fan_in = self.input_dim
            self.runtime_coef = self.gain / np.sqrt(fan_in)
            self.runtime_coef *= self.lr_multiplier

    def call(self, inputs):
        dtype = inputs.dtype.base_dtype.name
        if dtype != 'int32' and dtype != 'int64':
            inputs = tf.cast(inputs, 'int32')

        if self.use_weight_scaling:
            embeddings = self.embeddings * self.runtime_coef
        else:
            embeddings = self.embeddings
        if isinstance(embeddings, sharded_variable.ShardedVariable):
            out = tf.nn.embedding_lookup(embeddings.variables, inputs)
        else:
            out = tf.nn.embedding_lookup(embeddings, inputs)
        return out

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
