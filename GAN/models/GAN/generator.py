import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.model = None
        self.image_shape = (conf['input_size'], conf['input_size'], conf['channel'])
        self.build_model(input_dim=conf['latent_dim'],
                         size=conf['input_size'])

    def build_model(self, input_dim, size):
        model = [layers.Dense(128, activation=tf.nn.leaky_relu, input_dim=input_dim),
                 layers.Dense(256, activation=tf.nn.leaky_relu),
                 layers.Dense(256, activation=tf.nn.leaky_relu),
                 layers.Dense(size**2, activation=tf.nn.tanh)]
        self.model = tf.keras.Sequential(model, name='generator')

    def call(self, x, reshape=False):
        x = self.model(x)
        if reshape:
            return tf.reshape(x, (-1, *self.image_shape))
        return x
