import tensorflow as tf
import layers


class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.model = None
        self.build_model()

    def build_model(self):
        model = [layers.Linear(256, activation=tf.nn.leaky_relu),
                 layers.Linear(256, activation=tf.nn.leaky_relu),
                 layers.Linear(128, activation=tf.nn.leaky_relu),
                 layers.Linear(1)]
        self.model = tf.keras.Sequential(model, name='discriminator')

    def call(self, x):
        return self.model(x)
