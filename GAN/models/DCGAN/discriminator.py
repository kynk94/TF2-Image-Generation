import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self, conf):
        super(Discriminator, self).__init__()
        hp = conf['dis']
        self.model = None
        self.build_model(n_layer=hp['n_layer'],
                         n_filter=hp['n_filter'])

    def build_model(self, n_layer, n_filter):
        model = []
        for _ in range(n_layer):
            model.extend([layers.Conv2D(n_filter, (5, 5), strides=(2, 2),
                                        padding='same', activation=tf.nn.leaky_relu),
                          layers.BatchNormalization()])
            n_filter *= 2
        model.extend([layers.Flatten(),
                      layers.Dense(1)])
        self.model = tf.keras.Sequential(model, name='discriminator')

    def call(self, x):
        return self.model(x)
