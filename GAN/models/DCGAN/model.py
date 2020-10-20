import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models import BaseModel
from utils import tf_image_concat, tf_image_write
from .generator import Generator
from .discriminator import Discriminator


class DCGAN(BaseModel):
    def __init__(self, conf, use_log=True):
        super().__init__(conf)
        self.generator = Generator(conf)
        self.discriminator = Discriminator(conf)
        self.gen_opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.dis_opt = Adam(conf['learning_rate'], conf['beta_1'])

        self._bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._use_log = use_log
        self._latent_shape = (conf['batch_size'], conf['latent_dim'])
        self._set_checkpoint()
        if self._use_log:
            self._logger = self._create_logger()

    def train(self, x):
        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as d_tape:
            generated_image = self.generator(latent)
            real_score = self.discriminator(x)
            fake_score = self.discriminator(generated_image)
            loss_d = self._bce_loss(tf.ones_like(real_score), real_score)
            loss_d += self._bce_loss(tf.zeros_like(fake_score), fake_score)
        gradient_d = d_tape.gradient(loss_d,
                                     self.discriminator.trainable_variables)
        self.dis_opt.apply_gradients(zip(gradient_d,
                                         self.discriminator.trainable_variables))

        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as g_tape:
            generated_image = self.generator(latent)
            fake_score = self.discriminator(generated_image)
            loss_g = self._bce_loss(tf.ones_like(fake_score), fake_score)
        gradient_g = g_tape.gradient(loss_g,
                                     self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradient_g,
                                         self.generator.trainable_variables))

        if self._use_log:
            self._write_train_log(loss_g, loss_d)
        self.checkpoint.step.assign_add(1)
        return loss_g, loss_d

    def test(self, x, step=None, save=False, display_shape=None):
        generated_image = self.generator(x, training=False)
        if self._use_log:
            if display_shape is None:
                test_batch = x.shape[0]
                n_row = int(test_batch**0.5)
                display_shape = (n_row, n_row)
            concat_image = tf_image_concat(generated_image, display_shape)
            if save:
                tf_image_write(filename=os.path.join(self._output_dir,
                                                     '{:05d}.png'.format(step)),
                               contents=concat_image)
            self._write_test_log(step=step or self.checkpoint.step,
                                 data=tf.expand_dims(concat_image/2+0.5, axis=0))
        return generated_image

    def _set_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            generator_optimizer=self.gen_opt,
            discriminator_optimizer=self.dis_opt,
            generator=self.generator,
            discriminator=self.discriminator)

    def _write_train_log(self, loss_g, loss_d):
        step = self.checkpoint.step
        with self._logger.as_default():
            tf.summary.scalar(name='loss_gen', data=loss_g, step=step)
            tf.summary.scalar(name='loss_dis', data=loss_d, step=step)

    def _write_test_log(self, step, data):
        with self._logger.as_default():
            tf.summary.image(name='test_output', data=data, step=step)
