import tensorflow as tf
import tensorflow_addons as tfa
from .utils import get_layer_config


class Activation(tf.keras.layers.Activation):
    def __init__(self, activation, activation_alpha=0.3, **kwargs):
        super().__init__(
            activation=Activation.get_activation(activation, activation_alpha),
            **kwargs)

    @staticmethod
    def get_activation(activation, activation_alpha=0.3):
        if hasattr(activation, '__call__'):
            return activation
        if not isinstance(activation, str):
            raise ValueError(f'Unsupported `activation`: {activation}')
        l_activation = activation.lower()
        if l_activation in {'leaky_relu', 'lrelu'}:
            return tf.keras.layers.LeakyReLU(
                alpha=activation_alpha,
                name='leaky_relu')
        if l_activation in {'exp_lu', 'elu'}:
            return tf.keras.layers.ELU(
                alpha=activation_alpha,
                name='elu')
        if l_activation in {'trelu', 'tlu'}:
            return tfa.layers.TLU()
        return l_activation

    def get_config(self):
        config = super().get_config()
        config.update({
            'activation': get_layer_config(self.activation)
        })
        return config
