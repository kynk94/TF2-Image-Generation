import tensorflow as tf
import layers


class Discriminator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['dis']
        self.model = None
        if conf['use_residual']:
            self.build_residual_model(
                input_shape=(conf['channel'],
                             conf['input_size'],
                             conf['input_size']),
                n_layer=hp['n_layer'],
                n_filter=hp['n_filter'])
        else:
            self.build_model(
                input_shape=(conf['channel'],
                             conf['input_size'],
                             conf['input_size']),
                n_layer=hp['n_layer'],
                n_filter=hp['n_filter'])

    def build_residual_model(self, input_shape, n_layer, n_filter):
        model = [layers.Input(input_shape),
                 layers.Conv2D(n_filter, 3, 1, 1)]
        for _ in range(n_layer - 1):
            n_filter *= 2
            model.append(layers.DownResBlock2D(n_filter, 3, 1, 1,
                                               normalization='bn',
                                               activation='relu',
                                               normalization_first=True))
        model.extend([layers.Flatten(),
                      layers.Dense(1)])
        self.model = tf.keras.Sequential(model, name='discriminator')
        self.model.summary()

    def build_model(self, input_shape, n_layer, n_filter):
        model = [layers.Input(input_shape),
                 layers.Conv2DBlock(n_filter, 5, 2, 'same',
                                    activation='lrelu')]
        for _ in range(n_layer - 1):
            n_filter *= 2
            model.append(layers.Conv2DBlock(n_filter, 5, 2, 'same',
                                            normalization='bn',
                                            activation='lrelu'))
        model.extend([layers.Flatten(),
                      layers.Dense(1)])
        self.model = tf.keras.Sequential(model, name='discriminator')
        self.model.summary()

    def call(self, x):
        return self.model(x)
