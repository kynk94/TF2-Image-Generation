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
        size //= 2**n_layer
        model = [layers.Dense(n_filter*size*size, input_dim=input_dim),
                 layers.Reshape((n_filter, size, size)),
                 layers.BatchNormalization(axis=1),
                 layers.ReLU()]
        for _ in range(n_layer-1):
            n_filter //= 2
            model.extend([layers.Conv2DTranspose(n_filter, (5, 5), strides=(2, 2),
                                                 padding='same'),
                          layers.BatchNormalization(axis=1),
                          layers.ReLU()])
        model.append(layers.Conv2DTranspose(channel, (5, 5), strides=(2, 2),
                                            padding='same', activation=tf.nn.tanh))
        self.model = tf.keras.Sequential(model, name='generator')

    def call(self, x):
        return self.model(x)
