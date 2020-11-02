import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Sequential, layers


class Discriminator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['dis']

        block_label = [layers.Embedding(input_dim=conf['n_class'],
                                        output_dim=hp['hidden_dim_label']*hp['k_label'],
                                        embeddings_initializer='glorot_uniform'),
                       layers.ReLU(),
                       layers.Dropout(conf['dropout_rate']),
                       tfa.layers.Maxout(hp['hidden_dim_label'])]
        block_image = [layers.Dense(input_dim=conf['input_size']**2 * conf['channel'],
                                    units=hp['hidden_dim_image'],
                                    activation=tf.nn.relu),
                       layers.Dropout(conf['dropout_rate']),
                       tfa.layers.Maxout(hp['hidden_dim_image'])]
        block_combined = [layers.Dense(units=hp['hidden_dim_combined'],
                                       activation=tf.nn.relu),
                          layers.Dropout(conf['dropout_rate']),
                          tfa.layers.Maxout(hp['hidden_dim_combined']),
                          layers.Dense(1)]

        self.block_label = Sequential(block_label, name='block_label')
        self.block_image = Sequential(block_image, name='block_image')
        self.block_combined = Sequential(block_combined, name='block_combined')

    def call(self, image, label):
        image = self.block_image(image)
        label = self.block_label(label)
        out = self.block_combined(tf.concat([image, label], axis=-1))
        return out
