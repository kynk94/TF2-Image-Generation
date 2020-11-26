"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""


def get_layer_config(layer):
    if layer is None:
        return None
    if hasattr(layer, 'get_config'):
        return layer.get_config()
    return getattr(layer, '__name__',
                   layer.__class__.__name__)
