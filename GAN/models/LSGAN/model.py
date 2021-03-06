import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from layers import BaseModel
from utils import tf_image_concat
from .generator import Generator
from .discriminator import Discriminator


class LSGAN(BaseModel):
    def __init__(self, conf, ckpt=None):
        super().__init__(conf, ckpt)
        self.generator = Generator(conf)
        self.discriminator = Discriminator(conf)
        self.gen_opt = Adam(conf['gen']['learning_rate'], conf['beta_1'])
        self.dis_opt = Adam(conf['dis']['learning_rate'], conf['beta_1'])
        self.set_checkpoint(generator_optimizer=self.gen_opt,
                            discriminator_optimizer=self.dis_opt,
                            generator=self.generator,
                            discriminator=self.discriminator)

        self._mse_loss = tf.keras.losses.MeanSquaredError()
        self._latent_shape = (conf['batch_size'], conf['latent_dim'])

    @tf.function
    def train(self, x):
        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as d_tape:
            generated_image = self.generator(latent)
            score_d_real = self.discriminator(x)
            score_d_fake = self.discriminator(generated_image)
            loss_d = self._mse_loss(tf.ones_like(score_d_real), score_d_real)
            loss_d += self._mse_loss(tf.zeros_like(score_d_fake), score_d_fake)
            loss_d *= 0.5
        gradient_d = d_tape.gradient(loss_d,
                                     self.discriminator.trainable_variables)
        self.dis_opt.apply_gradients(zip(gradient_d,
                                         self.discriminator.trainable_variables))

        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as g_tape:
            generated_image = self.generator(latent)
            score_g_fake = self.discriminator(generated_image)
            loss_g = self._mse_loss(tf.ones_like(score_g_fake), score_g_fake)
            loss_g *= 0.5
        gradient_g = g_tape.gradient(loss_g,
                                     self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradient_g,
                                         self.generator.trainable_variables))

        log_dict = {
            'loss/gen': loss_g,
            'loss/dis': loss_d,
            'score/real': tf.reduce_mean(score_d_real),
            'score/fake': tf.reduce_mean(score_d_fake)
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
