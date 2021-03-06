import tensorflow as tf
import layers


class TransformNet(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['transform_net']
        self.model = None
        self.build_model(n_residual=hp['n_residual'],
                         n_filter=hp['n_filter'],
                         size=conf['input_size'],
                         channel=conf['channel'])

    def build_model(self, n_residual, n_filter, size, channel):
        model = [layers.InputLayer((channel, size, size)),
                 layers.Conv2DBlock(n_filter, 9, 1, 4,
                                    pad_type='reflect',
                                    normalization='in',
                                    activation='lrelu'),
                 layers.Conv2DBlock(n_filter*2, 3, 2, 1,
                                    pad_type='reflect',
                                    normalization='in',
                                    activation='lrelu'),
                 layers.Conv2DBlock(n_filter*4, 3, 2, 1,
                                    pad_type='reflect')]
        for _ in range(n_residual):
            model.append(layers.ResBlock2D(n_filter*4, 3, 1, 1,
                                           pad_type='reflect',
                                           normalization='in',
                                           activation='lrelu',
                                           normalization_first=True))
        model.extend([layers.Normalization('in'),
                      layers.ReLU(),
                      layers.TransConv2DBlock(n_filter*2, 3, 2, 'same',
                                              fir=[1, 2, 1],
                                              normalization='in',
                                              activation='lrelu'),
                      layers.TransConv2DBlock(n_filter, 3, 2, 'same',
                                              fir=[1, 2, 1],
                                              normalization='in',
                                              activation='lrelu'),
                      layers.Conv2DBlock(channel, 9, 1, 4,
                                         pad_type='reflect',
                                         normalization='in',
                                         activation='tanh')])
        self.model = tf.keras.Sequential(model, name='transform_net')
        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)
