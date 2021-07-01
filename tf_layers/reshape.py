from tensorflow.python.keras import layers as K_layers
from .utils import get_layer_config


class Reshape(K_layers.Reshape):
    def __init__(self, target_shape, perm_after_reshape=None, **kwargs):
        super().__init__(target_shape, **kwargs)
        if perm_after_reshape is None:
            self.perm_after_reshape = None
        else:
            self.perm_after_reshape = K_layers.Permute(perm_after_reshape)

    def call(self, inputs):
        outputs = super().call(inputs)
        if self.perm_after_reshape:
            return self.perm_after_reshape(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'perm_after_reshape': get_layer_config(self.perm_after_reshape)
        })
