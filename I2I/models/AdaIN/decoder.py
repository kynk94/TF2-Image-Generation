import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
import layers


class Decoder(tf.keras.Model):
    def __init__(self, conf, data_format=None):
        super().__init__()
        self.data_format = conv_utils.normalize_data_format(data_format)
        hp = conf['decoder']
        self.model = None
        self.build_model(n_filter=hp['n_filter'],
                         size=conf['input_size'],
                         channel=conf['channel'])

    def build_model(self, n_filter, size, channel):
        init_size = size // 2 ** 3
        init_shape = (init_size,) * 2
        if self.data_format == 'channels_first':
            init_shape = (n_filter * 2,) + init_shape
        else:
            init_shape += (n_filter * 2,)
        model = [layers.InputLayer(init_shape),
                 layers.Conv2D(n_filter, 3, 1, 1,
                               pad_type='reflect', activation='relu'),
                 layers.Upsample(2, method='zero'),
                 layers.FIRFilter([1, 2, 1])]
        for i in range(3):
            model.append(layers.Conv2D(n_filter, 3, 1, 1,
                                       pad_type='reflect', activation='relu'))
        for i in range(2):
            n_filter //= 2
            model.extend([layers.Conv2D(n_filter, 3, 1, 1,
                                        pad_type='reflect', activation='relu'),
                          layers.Upsample(2, method='zero'),
                          layers.FIRFilter([1, 2, 1]),
                          layers.Conv2D(n_filter, 3, 1, 1,
                                        pad_type='reflect', activation='relu')])
        model.append(layers.Conv2D(channel, 3, 1, 1,
                                   pad_type='reflect', activation='tanh'))
        self.model = tf.keras.Sequential(model, name='decoder')
        self.model.summary()

    def call(self, inputs):
        return self.model(inputs)
