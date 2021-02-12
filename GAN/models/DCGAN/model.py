import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from layers import BaseModel
from utils import tf_image_concat
from .generator import Generator
from .discriminator import Discriminator


class DCGAN(BaseModel):
    def __init__(self, conf, ckpt=None, strategy=None):
        super().__init__(conf, ckpt, strategy)
        self.model_init(conf)
        self._bce_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)
        self._latent_shape = (conf['batch_size'], conf['latent_dim'])

    @BaseModel.strategy
    def model_init(self, conf):
        self.generator = Generator(conf)
        self.discriminator = Discriminator(conf)
        self.gen_opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.dis_opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.set_checkpoint(generator_optimizer=self.gen_opt,
                            discriminator_optimizer=self.dis_opt,
                            generator=self.generator,
                            discriminator=self.discriminator)

    @tf.function
    def train(self, inputs):
        log_dict = self.train_step(inputs)
        self.write_scalar_log(**log_dict)
        self.ckpt.step.assign_add(1)
        return log_dict

    @BaseModel.strategy_run
    def train_step(self, inputs):
        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as d_tape:
            generated_image = self.generator(latent)
            score_d_real = self.discriminator(inputs)
            score_d_fake = self.discriminator(generated_image)
            loss_d = self._bce_loss(tf.ones_like(score_d_real), score_d_real)
            loss_d += self._bce_loss(tf.zeros_like(score_d_fake), score_d_fake)
        gradient_d = d_tape.gradient(loss_d,
                                     self.discriminator.trainable_variables)
        self.dis_opt.apply_gradients(zip(gradient_d,
                                         self.discriminator.trainable_variables))

        latent = tf.random.normal(shape=self._latent_shape)
        with tf.GradientTape() as g_tape:
            generated_image = self.generator(latent)
            score_g_fake = self.discriminator(generated_image)
            loss_g = self._bce_loss(tf.ones_like(score_g_fake), score_g_fake)
        gradient_g = g_tape.gradient(loss_g,
                                     self.generator.trainable_variables)
        self.gen_opt.apply_gradients(zip(gradient_g,
                                         self.generator.trainable_variables))

        return {
            'loss/gen': tf.reduce_mean(loss_g),
            'loss/dis': tf.reduce_mean(loss_d),
            'score/real': tf.reduce_mean(score_d_real),
            'score/fake': tf.reduce_mean(score_d_fake)
        }

    @tf.function
    def generate_image(self, inputs):
        return self.generator(inputs, training=False)

    def test(self, inputs, step=None, save=False, display_shape=None):
        if step is None:
            step = self.ckpt.step
        generated_image = self.generate_image(inputs)
        if display_shape is None:
            test_batch = inputs.shape[0]
            n_row = int(test_batch**0.5)
            display_shape = (n_row, n_row)

        concat_image = tf_image_concat(generated_image, display_shape)

        if save:
            self.image_write(filename='{:05d}.png'.format(step),
                             data=concat_image)
        self.write_image_log(step=step, data=concat_image)
        return generated_image
