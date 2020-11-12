import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models import BaseModel
from ops import calculate_gram_matrix
from .feature_extractor import FeatureExtractor


class NeuralStyleTransfer(BaseModel):
    def __init__(self, conf, init_image, content_image, style_image):
        super().__init__(conf)
        self.feature_extractor = FeatureExtractor(conf)
        self.opt = Adam(conf['learning_rate'], conf['beta_1'])
        self.set_checkpoint(optimizer=self.opt)

        self.init_image = tf.Variable(init_image)
        self.content_image = content_image
        self.style_image = style_image
        self.content_weight = conf['content_weight']

    @tf.function
    def train(self):
        with tf.GradientTape() as tape:
            content_drawing, style_drawing = self.feature_extractor(
                self.init_image)
            content_output, _ = self.feature_extractor(self.content_image)
            _, style_output = self.feature_extractor(self.style_image)

            content_loss = self.content_loss(content_drawing, content_output)
            style_loss = self.style_loss(style_drawing, style_output)
            loss = self.content_weight * content_loss + style_loss

        gradient = tape.gradient(loss, self.init_image)
        self.opt.apply_gradients([(gradient, self.init_image)])

        clipped_image = tf.clip_by_value(self.init_image, -1, 1)
        self.init_image.assign(clipped_image)

        log_dict = {
            'loss/content': content_loss,
            'loss/style': style_loss
        }
        self.write_scalar_log(**log_dict)
        self.ckpt.step.assign_add(1)
        return log_dict

    def write_init_image(self, init_shape, step=None, save=False):
        if step is None:
            step = self.ckpt.step

        for i, (image, shape) in enumerate(zip(self.init_image.numpy(), init_shape)):
            image = tf.transpose(image, (1, 2, 0))
            resized_image = tf.image.resize(image, shape)
            if save:
                self.image_write(filename=f'{i:03d}-{step:05d}.png',
                                 data=resized_image)
            self.write_image_log(step=step, data=resized_image,
                                 name=f'output/{i:03d}')

    def content_loss(self, drawing, content):
        out = 0
        for d, c in zip(drawing, content):
            out += tf.reduce_mean(tf.square(d - c))
        return 0.5 * out

    def style_loss(self, drawing, style):
        out = 0
        for d, s in zip(drawing, style):
            prod_shape = tf.cast(tf.reduce_prod(d.shape[1:]), dtype=tf.float32)
            scale_const = tf.square(prod_shape)
            matrix = calculate_gram_matrix(d) - calculate_gram_matrix(s)
            _out = tf.reduce_sum(tf.square(matrix)) / scale_const
            out += _out
        return 0.25 * out
