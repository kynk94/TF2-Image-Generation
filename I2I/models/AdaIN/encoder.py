import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input


class Encoder(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        hp = conf['encoder']
        self.model = None
        self.build_model(pretrained_model=hp['pretrained_model'],
                         output_layers=hp['output_layers'])

    def build_model(self, pretrained_model, output_layers):
        if pretrained_model.lower() == 'vgg19':
            from tensorflow.keras.applications import VGG19
            vgg = VGG19(include_top=False)
        elif pretrained_model.lower() == 'vgg16':
            from tensorflow.keras.applications import VGG16
            vgg = VGG16(include_top=False)
        else:
            raise ValueError(f'Unsupport {pretrained_model}')

        outputs = [vgg.get_layer(layer).output for layer in output_layers]
        self.model = tf.keras.Model(vgg.input, outputs, trainable=False)

    def call(self, inputs, denorm=False, clip=False):
        if denorm:
            inputs = inputs * 127.5 + 127.5
        if clip:
            inputs = tf.clip_by_value(inputs, 0, 255)
        # inputs = preprocess_input(inputs)
        return self.model(inputs)
