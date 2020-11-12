import tensorflow as tf
from tensorflow.keras import layers


class FeatureExtractor(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.model = None
        self.build_model(pretrained_model=conf['pretrained_model'],
                         content_layers=conf['content_layers'],
                         style_layers=conf['style_layers'])

    def build_model(self, pretrained_model, content_layers, style_layers):
        if pretrained_model.lower() == 'vgg19':
            from tensorflow.keras.applications import VGG19
            vgg = VGG19(include_top=False)
        elif pretrained_model.lower() == 'vgg16':
            from tensorflow.keras.applications import VGG16
            vgg = VGG16(include_top=False)
        else:
            raise ValueError(f'Unsupport {pretrained_model}')

        vgg.trainable = False
        content_output = [vgg.get_layer(layer).output
                          for layer in content_layers]
        style_output = [vgg.get_layer(layer).output
                        for layer in style_layers]
        self.model = tf.keras.Model(vgg.input, (content_output, style_output))

    def call(self, x):
        return self.model(x)
