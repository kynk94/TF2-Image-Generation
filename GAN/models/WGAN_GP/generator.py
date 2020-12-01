import tensorflow as tf
import layers


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
        kernel_size = size // 2**n_layer
        model = [layers.Input(input_dim),
                 layers.Reshape((input_dim, 1, 1)),
                 layers.TransConv2DBlock(n_filter, kernel_size, 1,
                                         normalization='bn',
                                         activation='relu')]
        for _ in range(n_layer-1):
            n_filter //= 2
            model.append(layers.TransConv2DBlock(n_filter, 5, 2,
                                                 conv_padding='same',
                                                 normalization='bn',
                                                 activation='relu'))
        model.append(layers.TransConv2DBlock(channel, 5, 2,
                                             conv_padding='same',
                                             activation='tanh'))
        self.model = tf.keras.Sequential(model, name='generator')
        self.model.summary()

    def call(self, x):
        return self.model(x)
