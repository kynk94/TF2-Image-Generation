import tensorflow as tf
import layers

class Generator(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['gen']
        self.image_shape = (conf['channel'],
                            conf['input_size'],
                            conf['input_size'])

        block_label = [layers.Embedding(input_dim=conf['n_class'],
                                        output_dim=hp['hidden_dim_label']),
                       layers.ReLU(),
                       layers.Dropout(conf['dropout_rate'])]
        block_latent = [layers.Dense(input_dim=conf['latent_dim'],
                                     units=hp['hidden_dim_latent'],
                                     activation=tf.nn.relu),
                        layers.Dropout(conf['dropout_rate'])]
        block_combined = [layers.Dense(units=hp['hidden_dim_combined'],
                                       activation=tf.nn.relu),
                          layers.Dropout(conf['dropout_rate'])]

        self.block_label = tf.keras.Sequential(block_label, name='block_label')
        self.block_latent = tf.keras.Sequential(block_latent, name='block_latent')
        self.block_combined = tf.keras.Sequential(block_combined, name='block_combined')
        self.dense_last = layers.Dense(units=conf['input_size']**2 * conf['channel'],
                                       activation=tf.nn.tanh,
                                       name='dense_last')

    def call(self, latent, label, reshape=False):
        latent = self.block_latent(latent)
        label = self.block_label(label)
        out = self.block_combined(tf.concat([latent, label], axis=-1))
        out = self.dense_last(out)
        if reshape:
            return tf.reshape(out, (-1, *self.image_shape))
        return out
