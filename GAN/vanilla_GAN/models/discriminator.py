import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = None
        self.build_model()

    def build_model(self):
        model = [layers.Dense(256, activation=tf.nn.leaky_relu),
                 layers.Dense(256, activation=tf.nn.leaky_relu),
                 layers.Dense(128, activation=tf.nn.leaky_relu),
                 layers.Dense(1)]
        self.model = tf.keras.Sequential(model, name='discriminator')

    def call(self, x):
        return self.model(x)
