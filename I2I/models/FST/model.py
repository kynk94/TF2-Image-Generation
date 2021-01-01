import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from layers import BaseModel
from utils import tf_image_concat
from ops import calculate_gram_matrix
from .feature_extractor import FeatureExtractor
from .transform_net import TransformNet


class FastStyleTransfer(BaseModel):
    def __init__(self, conf, style_image, ckpt=None):
        super().__init__(conf, ckpt)
        self.feature_extractor = FeatureExtractor(conf['feature_extrator'])
        self.transform_net = TransformNet(conf)
        self.opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.set_checkpoint(transform_net=self.transform_net,
                            optimizer=self.opt)

        self.style_target = self.feature_extractor(style_image)[1]
        self.content_weight = conf['content_weight']
        self.total_variation_weight = conf['total_variation_weight']

    @tf.function
    def train(self, inputs):
        with tf.GradientTape() as tape:
            generated_image = self.transform_net(inputs)
            content_feature, style_feature = self.feature_extractor(
                generated_image)
            content_target = self.feature_extractor(inputs)[0]
            content_loss = self.content_loss(content_feature, content_target)
            content_loss *= self.content_weight
            style_loss = self.style_loss(style_feature, self.style_target) * 100
            total_variation_loss = self.total_variation_loss(generated_image)
            total_variation_loss *= self.total_variation_weight
            loss = content_loss + style_loss + total_variation_loss
        gradient = tape.gradient(loss, self.transform_net.trainable_variables)
        self.opt.apply_gradients(zip(gradient,
                                     self.transform_net.trainable_variables))

        log_dict = {
            'loss/content': content_loss,
            'loss/style': style_loss,
            'loss/total_variation': total_variation_loss
        }
        self.write_scalar_log(**log_dict)
        self.ckpt.step.assign_add(1)
        return log_dict

    def test(self, inputs, step=None, save=False, save_input=False, display_shape=None):
        if step is None:
            step = self.ckpt.step
        generated_image = self.transform_net(inputs, training=False)
        if display_shape is None:
            test_batch = inputs.shape[0]
            n_row = int(test_batch**0.5)
            display_shape = (n_row, n_row)

        n_display = display_shape[0] * display_shape[1]
        concat_image = tf_image_concat(
            generated_image[:n_display], display_shape)

        if save:
            self.image_write(filename=f'{step:05d}.png', data=concat_image)
        if save_input:
            inputs_data = tf_image_concat(inputs[:n_display], display_shape)
            self.image_write(filename='inputs.png', data=inputs_data)
            self.write_image_log(step=step, data=inputs_data, name='inputs')
        self.write_image_log(step=step, data=concat_image)
        return generated_image

    def content_loss(self, inputs, content):
        out = 0
        for i, c in zip(inputs, content):
            out += tf.reduce_mean(tf.square(i - c))
        return 0.5 * out

    def style_loss(self, inputs, style):
        out = 0
        for i, s in zip(inputs, style):
            prod_shape = tf.cast(tf.reduce_prod(i.shape[1:]), dtype=tf.float32)
            scale_const = tf.square(prod_shape)
            matrix = calculate_gram_matrix(i) - calculate_gram_matrix(s)
            out += tf.reduce_sum(tf.square(matrix)) / scale_const
        return 0.25 * out

    def total_variation_loss(self, inputs):
        h_var = inputs[..., 1:, :] - inputs[..., :-1, :]
        w_var = inputs[..., 1:] - inputs[..., :-1]
        h = tf.reduce_mean(tf.square(h_var))
        w = tf.reduce_mean(tf.square(w_var))
        return h + w
