import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from layers import BaseModel
from utils import tf_image_concat
from .generator import Generator
from .discriminator import Discriminator


class WGAN_GP(BaseModel):
    def __init__(self, conf, ckpt=None):
        super().__init__(conf, ckpt)
        self.generator = Generator(conf)
        self.discriminator = Discriminator(conf)
        self.gen_opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.dis_opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.set_checkpoint(generator_optimizer=self.gen_opt,
                            discriminator_optimizer=self.dis_opt,
                            generator=self.generator,
                            discriminator=self.discriminator)

        self.penalty_lambda = conf['penalty_lambda']
        self._latent_shape = (conf['batch_size'], conf['latent_dim'])

    @tf.function
    def train_generator(self):
        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as g_tape:
            generated_image = self.generator(latent)
            score_g_fake = self.discriminator(generated_image)
            loss_g = -tf.reduce_mean(score_g_fake)
        gradient_g = g_tape.gradient(loss_g,
                                     self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradient_g,
                                         self.generator.trainable_variables))

        log_dict = {'loss/gen': loss_g}
        self.write_scalar_log(**log_dict)
        return log_dict

    @tf.function
    def train_discriminator(self, x):
        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as d_tape:
            generated_image = self.generator(latent)
            score_d_real = self.discriminator(x)
            score_d_fake = self.discriminator(generated_image)
            penalty, score_d_interpolation = \
                self.gradient_penalty(x, generated_image)
            penalty *= self.penalty_lambda

            loss_d = tf.reduce_mean(score_d_fake)
            loss_d -= tf.reduce_mean(score_d_real)
            loss_d += penalty
        gradient_d = d_tape.gradient(loss_d,
                                     self.discriminator.trainable_variables)
        self.dis_opt.apply_gradients(zip(gradient_d,
                                         self.discriminator.trainable_variables))

        log_dict = {
            'loss/dis': loss_d,
            'loss/penalty': penalty,
            'score/real': tf.reduce_mean(score_d_real),
            'score/fake': tf.reduce_mean(score_d_fake),
            'score/interpolation': tf.reduce_mean(score_d_interpolation)
        }
        self.write_scalar_log(**log_dict)
        self.ckpt.step.assign_add(1)
        return log_dict

    def test(self, x, step=None, save=False, display_shape=None):
        if step is None:
            step = self.ckpt.step
        generated_image = self.generator(x, training=False)
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

    def gradient_penalty(self, real_image, fake_image):
        epsilon = tf.random.uniform((real_image.shape[0], 1, 1, 1),
                                    minval=0.0, maxval=1.0)
        interpolation = epsilon * real_image + (1 - epsilon) * fake_image

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolation)
            score_d_interpolation = self.discriminator(interpolation)
        gradient = gp_tape.gradient(score_d_interpolation,
                                    interpolation)
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=(1, 2, 3)))
        penalty = tf.reduce_mean((norm - 1.0)**2)
        return penalty, score_d_interpolation
