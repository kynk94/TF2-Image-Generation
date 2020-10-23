import tensorflow as tf
from tensorflow.keras import layers


class Generator(tf.keras.Model):
    def __init__(self, conf):
        super(Generator, self).__init__()
        hp = conf['gen']
        self.model = None
        self.build_model(n_layer=hp['n_layer'],
                         n_filter=hp['n_filter'],
                         size=conf['input_size'],
                         channel=conf['channel'])

    def build_model(self, n_layer, n_filter, size, channel):
        size //= 2**n_layer
        model = [layers.Dense(size*size*n_filter[0]),
                 layers.BatchNormalization(),
                 layers.LeakyReLU(),
                 layers.Reshape((size, size, n_filter[0]))]
        for i in range(1, n_layer):
            model.extend([layers.Conv2DTranspose(n_filter[i], (5, 5), strides=(2, 2),
                                                 padding='same'),
                          layers.BatchNormalization(),
                          layers.LeakyReLU()])
        model.append(layers.Conv2DTranspose(channel, (5, 5), strides=(2, 2),
                                            padding='same', activation=tf.nn.tanh))
        self.model = tf.keras.Sequential(model, name='generator')

    def call(self, x):
        return self.model(x)
