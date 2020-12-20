import tensorflow as tf
import layers


class Generator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['gen']
        self.model = None
        if conf['use_residual']:
            self.build_residual_model(
                input_dim=conf['latent_dim'],
                n_layer=hp['n_layer'],
                n_filter=hp['n_filter'],
                size=conf['input_size'],
                channel=conf['channel'])
        else:
            self.build_model(
                input_dim=conf['latent_dim'],
                n_layer=hp['n_layer'],
                n_filter=hp['n_filter'],
                size=conf['input_size'],
                channel=conf['channel'])

    def build_residual_model(self, input_dim, n_layer, n_filter, size, channel):
        init_size = size // 2**(n_layer - 1)
        model = [layers.Input(input_dim),
                 layers.Linear(n_filter * init_size**2),
                 layers.Reshape((n_filter, init_size, init_size))]
        for _ in range(n_layer - 1):
            model.append(layers.UpResBlock2D(n_filter, 3, 1, 1,
                                             normalization='bn',
                                             activation='relu',
                                             normalization_first=True))
            n_filter //= 2
        model.extend([layers.Conv2DBlock(channel, 3, 1, 1,
                                         activation='relu',
                                         normalization='bn',
                                         normalization_first=True),
                      layers.Activation('tanh')])
        self.model = tf.keras.Sequential(model, name='generator')
        self.model.summary()

    def build_model(self, input_dim, n_layer, n_filter, size, channel):
        init_size = size // 2**n_layer
        model = [layers.Input(input_dim),
                 layers.LinearBlock(n_filter * init_size**2,
                                    normalization='bn',
                                    activation='relu'),
                 layers.Reshape((n_filter, init_size, init_size))]
        for _ in range(n_layer - 1):
            n_filter //= 2
            model.append(layers.TransConv2DBlock(n_filter, 5, 2, 'same',
                                                 normalization='bn',
                                                 activation='relu'))
        model.append(layers.TransConv2DBlock(channel, 5, 2, 'same',
                                             activation='tanh'))
        self.model = tf.keras.Sequential(model, name='generator')
        self.model.summary()

    def call(self, x):
        return self.model(x)
