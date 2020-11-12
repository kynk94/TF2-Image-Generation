import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['gen']
        self.model = None
        self.build_model(input_dim=conf['latent_dim'],
                         n_layer=hp['n_layer'],
                         n_filter=hp['n_filter'],
                         size=conf['input_size'],
                         channel=conf['channel'])

    def build_model(self, input_dim, n_layer, n_filter, size, channel):
        size //= 2**(n_layer - 3)
        model = [layers.Dense(n_filter*size*size, input_dim=input_dim),
                 layers.Reshape((n_filter, size, size)),
                 layers.BatchNormalization(),
                 layers.LeakyReLU()]
        for _ in range(2):
            model.extend([layers.Conv2DTranspose(n_filter, (3, 3), strides=(2, 2),
                                                 padding='same'),
                          layers.BatchNormalization(),
                          layers.LeakyReLU(),
                          layers.Conv2DTranspose(n_filter, (3, 3), strides=(1, 1),
                                                 padding='same'),
                          layers.BatchNormalization(),
                          layers.LeakyReLU()])
        for _ in range(n_layer - 5):
            n_filter //= 2
            model.extend([layers.Conv2DTranspose(n_filter, (3, 3), strides=(2, 2),
                                                 padding='same'),
                          layers.BatchNormalization(),
                          layers.LeakyReLU()])
        model.append(layers.Conv2DTranspose(channel, (3, 3), strides=(1, 1),
                                            padding='same', activation=tf.nn.tanh))
        self.model = tf.keras.Sequential(model, name='generator')

    def call(self, x):
        return self.model(x)
