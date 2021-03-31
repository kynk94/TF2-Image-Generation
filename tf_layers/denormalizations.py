import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.utils import conv_utils
from .conv import Conv
from .linear import Linear
from .normalizations import Normalization
from .resample import Resample
from .reshape import Reshape
from .utils import get_layer_config


class Denormalization(tf.keras.layers.Layer):
    """
    Denormalization Layer

    Arguments:
        normalization: Type of Parameter-free normalization for inputs.
            (e.g. batch_norm, sync_batch_norm, instance_norm)
        epsilon: Small float added to variance to avoid dividing by zero.
        data_format: A string, one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch_size, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch_size, channels, ...)`.
        trainable: Boolean, if `True` the weights of this layer will be marked as
            trainable (and listed in `layer.trainable_weights`).
        name: A string, the name of the layer.
    """

    def __init__(self,
                 normalization='bn',
                 epsilon=1e-5,
                 data_format=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(trainable=trainable, name=name, **kwargs)
        self.normalization = normalization
        self.epsilon = epsilon
        self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.rank = len(input_shape) - 2
        self._channel_axis = self._get_channel_axis()
        self._spatial_axes = self._get_spatial_axes()
        self.normalization = Normalization(
            normalization=self.normalization,
            epsilon=self.epsilon,
            center=False,
            scale=False,
            data_format=self.data_format,
            name='normalization')
        super().build(input_shape)

    def call(self, inputs, scale, offset, alpha=1.0):
        normalized_inputs = self.normalization(inputs)
        outputs = normalized_inputs * scale + offset
        if alpha == 1.0:
            return outputs
        return alpha * normalized_inputs + (1.0 - alpha) * inputs

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        return self.rank + 1

    def _get_spatial_axes(self):
        channel_axis = self._get_channel_axis()
        spatial_axes = list(range(self.rank + 2))
        del spatial_axes[channel_axis]
        del spatial_axes[0]
        return spatial_axes

    def get_config(self):
        config = super().get_config()
        config.update({
            'normalization': get_layer_config(self.normalization)
        })
        return config


class AdaIN(Denormalization):
    def __init__(self,
                 epsilon=1e-5,
                 data_format=None,
                 use_learnable_params=False,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            normalization='in',
            epsilon=epsilon,
            data_format=data_format,
            trainable=trainable,
            name=name,
            **kwargs)
        self.use_learnable_params = use_learnable_params

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[self._channel_axis]
        if self.use_learnable_params:
            if self.data_format == 'channels_first':
                perm_after_reshape = None
            else:
                perm_after_reshape = (*range(2, self.rank+2), 1)
            out_dims = 2 * input_channel
            self.linear = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                Linear(out_dims),
                Reshape(target_shape=(out_dims,) + (1,)*self.rank,
                        perm_after_reshape=perm_after_reshape)],
                name='linear')

    def call(self, inputs, external_inputs, alpha=1.0):
        if self.use_learnable_params:
            adain_params = self.linear(external_inputs)
            gamma, beta = tf.split(adain_params, 2, self._channel_axis)
        else:
            gamma, beta = tf.nn.moments(external_inputs,
                                        self._spatial_axes,
                                        keepdims=True)
        return super().call(
            inputs=inputs,
            scale=gamma,
            offset=beta,
            alpha=alpha)


class SPADE(Denormalization):
    def __init__(self,
                 filters=128,
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 activation='relu',
                 use_sync_norm=False,
                 use_weight_scaling=False,
                 gain=np.sqrt(2),
                 lr_multiplier=1.0,
                 epsilon=1e-5,
                 data_format=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super().__init__(
            normalization='sync_bn' if use_sync_norm else 'bn',
            epsilon=epsilon,
            data_format=data_format,
            trainable=trainable,
            name=name,
            **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_weight_scaling = use_weight_scaling
        self.gain = gain
        self.lr_multiplier = lr_multiplier

    def build(self, input_shape):
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        input_channel = input_shape[self._channel_axis]
        spatial_dims = [input_shape[axis] for axis in self._spatial_axes]
        self.resample = Resample(
            size=spatial_dims,
            method='nearest')
        self.mlp_shared = Conv(
            rank=self.rank,
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            use_weight_scaling=self.use_weight_scaling,
            gain=self.gain,
            lr_multiplier=self.lr_multiplier)
        self.mlp_gamma = Conv(
            rank=self.rank,
            filters=input_channel,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_weight_scaling=self.use_weight_scaling,
            gain=self.gain,
            lr_multiplier=self.lr_multiplier)
        self.mlp_beta = Conv(
            rank=self.rank,
            filters=input_channel,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_weight_scaling=self.use_weight_scaling,
            gain=self.gain,
            lr_multiplier=self.lr_multiplier)

    def call(self, inputs, external_inputs, alpha=1.0):
        external_inputs = self.resample(external_inputs)
        activations = self.mlp_shared(external_inputs)
        gamma = self.mlp_gamma(activations) + 1.0
        beta = self.mlp_beta(activations)
        return super().call(
            inputs=inputs,
            scale=gamma,
            offset=beta,
            alpha=alpha)
