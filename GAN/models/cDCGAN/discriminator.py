import tensorflow as tf
import layers


class Discriminator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['dis']
        self.model = None
        self.build_model(input_shape=((conf['channel'] +
                                       conf['label_dim'] * conf['n_class']),
                                      conf['input_size'],
                                      conf['input_size']),
                         n_layer=hp['n_layer'],
                         n_filter=hp['n_filter'])

    def build_model(self, input_shape, n_layer, n_filter):
        model = [layers.Input(input_shape),
                 layers.Conv2DBlock(filters=n_filter, kernel_size=5,
                                    strides=2, conv_padding='same',
                                    activation='lrelu')]
        for _ in range(n_layer - 1):
            n_filter *= 2
            model.extend([layers.Conv2DBlock(filters=n_filter, kernel_size=5,
                                             strides=2, conv_padding='same',
                                             normalization='bn',
                                             activation='lrelu')])
        model.extend([layers.Flatten(),
                      layers.Dense(1)])
        self.model = tf.keras.Sequential(model, name='discriminator')
        self.model.summary()

    def call(self, images, labels):
        batch, _, h, w = images.shape
        labels = tf.reshape(labels, (batch, -1, 1, 1))
        labels = tf.tile(labels, (1, 1, h, w))
        inputs = tf.concat((images, labels), axis=1)
        return self.model(inputs)
