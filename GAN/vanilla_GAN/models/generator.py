import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self, conf):
        super(Generator, self).__init__()
        self.model = None
        self.build_model(size=conf['size'])

    def build_model(self, size):
        model = [layers.Dense(128, activation=tf.nn.leaky_relu),
                 layers.Dense(256, activation=tf.nn.leaky_relu),
                 layers.Dense(256, activation=tf.nn.leaky_relu),
                 layers.Dense(size**2, activation=tf.nn.tanh)]
        self.model = tf.keras.Sequential(model, name='generator')

    def call(self, x):
        return self.model(x)
