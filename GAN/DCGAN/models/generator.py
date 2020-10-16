import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self, conf):
        super(Generator, self).__init__()
        hp = conf['gen']
        self.model = None
        self.build_model(n_layer=hp['n_layer'],
                         n_filter=hp['n_filter'],
                         size=conf['size'],
                         channel=conf['channel'])

    def build_model(self, n_layer, n_filter, size, channel):
        size //= 2**n_layer
        model = [layers.Dense(size*size*n_filter, use_bias=False),
                 layers.BatchNormalization(),
                 layers.LeakyReLU(),
                 layers.Reshape((size, size, n_filter))]
        for _ in range(n_layer - 1):
            n_filter //= 2
            model.extend([layers.Conv2DTranspose(n_filter//2, (5, 5), strides=(2, 2),
                                                 padding='same', use_bias=False),
                          layers.BatchNormalization(),
                          layers.LeakyReLU()])
        model.append(layers.Conv2DTranspose(channel, (5, 5), strides=(2, 2),
                                            padding='same', use_bias=False,
                                            activation=tf.nn.tanh))
        self.model = tf.keras.Sequential(model, name='generator')

    def call(self, x):
        return self.model(x)
