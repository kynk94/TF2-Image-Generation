import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models import BaseModel
from utils import tf_image_concat, tf_image_write
from .generator import Generator
from .discriminator import Discriminator


class GAN(BaseModel):
    def __init__(self, conf, ckpt=None):
        super().__init__(conf, ckpt)
        self.generator = Generator(conf)
        self.discriminator = Discriminator()
        self.gen_opt = Adam(conf['learning_rate'])
        self.dis_opt = Adam(conf['learning_rate'])
        self.set_checkpoint(generator_optimizer=self.gen_opt,
                            discriminator_optimizer=self.dis_opt,
                            generator=self.generator,
                            discriminator=self.discriminator)

        self._bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self._latent_shape = (conf['batch_size'], conf['latent_dim'])

    def train(self, x):
        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            generated_image = self.generator(latent)
            score_real = self.discriminator(x)
            score_fake = self.discriminator(generated_image)
            loss_g = self._bce_loss(tf.ones_like(score_fake), score_fake)
            loss_d = self._bce_loss(tf.ones_like(score_real), score_real)
            loss_d += self._bce_loss(tf.zeros_like(score_fake), score_fake)

        gradient_g = g_tape.gradient(loss_g,
                                     self.generator.trainable_variables)
        gradient_d = d_tape.gradient(loss_d,
                                     self.discriminator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradient_g,
                                         self.generator.trainable_variables))
        self.dis_opt.apply_gradients(zip(gradient_d,
                                         self.discriminator.trainable_variables))

        log_dict = {
            'loss/gen': loss_g,
            'loss/dis': loss_d,
            'score/real': tf.reduce_mean(score_real),
            'score/fake': tf.reduce_mean(score_fake)
        }
        self.write_scalar_log(**log_dict)
        self.ckpt.step.assign_add(1)
        return log_dict

    def test(self, x, step=None, save=False, display_shape=None):
        if step is None:
            step = self.ckpt.step
        generated_image = self.generator(x, reshape=True, training=False)
        if display_shape is None:
            test_batch = x.shape[0]
            n_row = int(test_batch**0.5)
            display_shape = (n_row, n_row)

        concat_image = tf_image_concat(generated_image, display_shape)

        if save:
            self.image_write(filename='{:05d}.png'.format(step),
                             data=concat_image)
        self.write_image_log(step=step, data=concat_image)
        return generated_image
