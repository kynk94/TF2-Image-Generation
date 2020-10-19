import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from utils import tf_image_concat
from .generator import Generator
from .discriminator import Discriminator


class DCGAN():
    def __init__(self, conf, use_log=True):
        self.step = 0
        self.generator = Generator(conf)
        self.discriminator = Discriminator(conf)
        self.gen_opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.dis_opt = Adam(conf['learning_rate'], conf['beta_1'])

        self._bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._use_log = use_log
        self._batch_size = conf['batch_size']
        self._latent_dim = conf['latent_dim']
        self._random_seed = conf['random_seed']
        self._set_checkpoint(conf['checkpoint_dir'])
        if self._use_log:
            self._create_logger(conf['log_dir'])

    def train(self, x):
        latent = tf.random.normal(shape=(self._batch_size, self._latent_dim),
                                  seed=self._random_seed)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_image = self.generator(latent, training=True)

            real_score = self.discriminator(x, training=True)
            fake_score = self.discriminator(generated_image, training=True)

            loss_g = self._bce_loss(tf.ones_like(fake_score), fake_score)
            loss_d = self._bce_loss(tf.ones_like(real_score), real_score)
            loss_d += self._bce_loss(tf.zeros_like(fake_score), fake_score)

        gradient_g = g_tape.gradient(loss_g,
                                     self.generator.trainable_variables)
        gradient_d = d_tape.gradient(loss_d,
                                     self.discriminator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradient_g,
                                         self.generator.trainable_variables))
        self.dis_opt.apply_gradients(zip(gradient_d,
                                         self.discriminator.trainable_variables))

        if self._use_log:
            self._write_train_log(self.step, loss_g, loss_d)
        self.step += 1
        return loss_g, loss_d

    def test(self, x, step=None, display_shape=None):
        generated_image = self.generator(x, training=False) / 2 + 0.5
        if self._use_log:
            if display_shape is None:
                test_batch = x.shape[0]
                n_row = int(test_batch**0.5)
                display_shape = (n_row, n_row)
            data = tf.expand_dims(tf_image_concat(generated_image, display_shape),
                                  axis=0)
            self._write_test_log(step=step or self.step,
                                 data=data)
        return generated_image

    def _set_checkpoint(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.gen_opt,
            discriminator_optimizer=self.dis_opt,
            generator=self.generator,
            discriminator=self.discriminator)

    def save(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

    def load(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.checkpoint.restore(checkpoint_dir)

    def _create_logger(self, log_dir):
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self._logger = tf.summary.create_file_writer(
            os.path.join(log_dir, f'DCGAN_{time_stamp}'))

    def _write_train_log(self, step, loss_g, loss_d):
        with self._logger.as_default():
            tf.summary.scalar(name='loss_gen', data=loss_g, step=step)
            tf.summary.scalar(name='loss_dis', data=loss_d, step=step)

    def _write_test_log(self, step, data):
        with self._logger.as_default():
            tf.summary.image(name='test_output', data=data, step=step)
