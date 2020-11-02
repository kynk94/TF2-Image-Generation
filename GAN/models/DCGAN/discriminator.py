import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['dis']
        self.model = None
        self.build_model(input_shape=(conf['input_size'],
                                      conf['input_size'],
                                      conf['channel']),
                         n_layer=hp['n_layer'],
                         n_filter=hp['n_filter'])

    def build_model(self, input_shape, n_layer, n_filter):
        model = [layers.Conv2D(n_filter, (5, 5), strides=(2, 2),
                               padding='same', input_shape=input_shape),
                 layers.BatchNormalization(),
                 layers.LeakyReLU()]
        for _ in range(n_layer - 1):
            n_filter *= 2
            model.extend([layers.Conv2D(n_filter, (5, 5), strides=(2, 2),
                                        padding='same'),
                          layers.BatchNormalization(),
                          layers.LeakyReLU()])
        model.extend([layers.Flatten(),
                      layers.Dense(1)])
        self.model = tf.keras.Sequential(model, name='discriminator')

    def call(self, x):
        return self.model(x)
