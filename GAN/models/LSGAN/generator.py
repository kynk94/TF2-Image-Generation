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
        init_size = size // 2**(n_layer - 3)
        model = [layers.Input(input_dim),
                 layers.LinearBlock(n_filter * init_size**2,
                                    normalization='bn',
                                    activation='relu'),
                 layers.Reshape((n_filter, init_size, init_size))]
        for _ in range(2):
            model.extend([layers.TransConv2DBlock(n_filter, 3, 2, 'same',
                                                  normalization='bn',
                                                  activation='lrelu'),
                          layers.TransConv2DBlock(n_filter, 3, 1, 'same',
                                                  normalization='bn',
                                                  activation='lrelu')])
        for _ in range(n_layer - 5):
            n_filter //= 2
            model.append(layers.TransConv2DBlock(n_filter, 3, 2, 'same',
                                                 normalization='bn',
                                                 activation='lrelu'))
        model.append(layers.TransConv2DBlock(channel, 3, 1, 'same',
                                             activation='tanh'))
        self.model = tf.keras.Sequential(model, name='generator')
        self.model.summary()

    def call(self, x):
        return self.model(x)
