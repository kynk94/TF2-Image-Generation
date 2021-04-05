import tensorflow as tf
from tensorflow.keras.optimizers import Adam

import layers
from layers import BaseModel
from utils import tf_image_concat
from .encoder import Encoder
from .decoder import Decoder


class AdaIN(BaseModel):
    def __init__(self, conf, ckpt=None):
        super().__init__(conf, ckpt)
        self.encoder = Encoder(conf)
        self.decoder = Decoder(conf)
        self.adain = layers.AdaIN()
        self.opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.set_checkpoint(decoder=self.decoder,
                            optimizer=self.opt)

    @tf.function
    def train(self, inputs):
        content_images, style_images = inputs
        with tf.GradientTape() as tape:
            content_feature = self.encoder(content_images)[-1]
            style_features = self.encoder(style_images)
            adain_outputs = self.adain(content_feature, style_features[-1])
            generated_images = self.decoder(adain_outputs)
            generated_features = self.encoder(generated_images)

            content_loss = self.content_loss([adain_outputs],
                                             [generated_features[-1]]) * 0.1
            style_loss = self.style_loss(style_features, generated_features)
            loss = content_loss + style_loss
        gradient = tape.gradient(loss, self.decoder.trainable_variables)
        self.opt.apply_gradients(zip(gradient,
                                     self.decoder.trainable_variables))

        log_dict = {
            'loss/content': content_loss,
            'loss/style': style_loss
        }
        self.write_scalar_log(**log_dict)
        self.ckpt.step.assign_add(1)
        return log_dict

    @tf.function
    def generate_image(self,
                       content_images,
                       style_images,
                       alpha=1.0,
                       training=False):
        content_feature = self.encoder(content_images, training=training)[-1]
        style_feature = self.encoder(style_images, training=training)[-1]
        adain_outputs = self.adain(content_feature,
                                   style_feature,
                                   alpha=alpha,
                                   training=training)
        return self.decoder(adain_outputs, training=training)

    def test(self, inputs, step=None, save=False, save_input=False, display_shape=None):
        content_images, style_images = inputs
        if step is None:
            step = self.ckpt.step

        generated_image = self.generate_image(
            content_images, style_images, training=False)
        if display_shape is None:
            test_batch = content_images.shape[0]
            n_row = int(test_batch**0.5)
            display_shape = (n_row, n_row)

        n_display = display_shape[0] * display_shape[1]
        concat_image = tf_image_concat(
            generated_image[:n_display], display_shape)

        if save:
            self.image_write(filename=f'{step:05d}.png', data=concat_image)
        if save_input:
            inputs_data = tf_image_concat(content_images[:n_display],
                                          display_shape)
            self.image_write(filename='contents.png', data=inputs_data)
            self.write_image_log(step=step, data=inputs_data,
                                 name='content_inputs')
            inputs_data = tf_image_concat(style_images[:n_display],
                                          display_shape)
            self.image_write(filename='styles.png', data=inputs_data)
            self.write_image_log(step=step, data=inputs_data,
                                 name='style_inputs')
        self.write_image_log(step=step, data=concat_image)
        return generated_image

    def content_loss(self, input_features, target_features):
        out = 0.0
        for i, t in zip(input_features, target_features):
            out += tf.reduce_mean(tf.square(i - t), axis=(1, 2, 3)) 
        return tf.reduce_mean(out)

    def style_loss(self, input_features, target_features, epsilon=1e-5):
        out = 0.0
        for i, t in zip(input_features, target_features):
            input_mean, input_var = tf.nn.moments(i, self.adain._spatial_axes, keepdims=True)
            target_mean, target_var = tf.nn.moments(t, self.adain._spatial_axes, keepdims=True)
            input_std = tf.sqrt(input_var + epsilon)
            target_std = tf.sqrt(target_var + epsilon)
            out += tf.reduce_mean(tf.square(input_mean - target_mean), axis=(1, 2, 3))
            out += tf.reduce_mean(tf.square(input_std - target_std), axis=(1, 2, 3))
        return tf.reduce_mean(out)
