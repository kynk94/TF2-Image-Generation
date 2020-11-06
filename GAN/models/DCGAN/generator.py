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
        model = [layers.Dense(size*size*n_filter, input_dim=input_dim),
                 layers.BatchNormalization(),
                 layers.ReLU(),
                 layers.Reshape((n_filter, size, size))]
        for _ in range(n_layer-1):
            n_filter //= 2
            model.extend([layers.Conv2DTranspose(n_filter, (5, 5), strides=(2, 2),
                                                 padding='same', data_format='channels_first'),
                          layers.BatchNormalization(),
                          layers.ReLU()])
        model.append(layers.Conv2DTranspose(channel, (5, 5), strides=(2, 2),
                                            padding='same', activation=tf.nn.tanh,
                                            data_format='channels_first'))
        self.model = tf.keras.Sequential(model, name='generator')

    def call(self, x):
        return self.model(x)
