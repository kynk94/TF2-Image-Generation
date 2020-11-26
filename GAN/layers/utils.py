def get_layer_config(layer):
    if layer is None:
        return None
    if hasattr(layer, 'get_config'):
        return layer.get_config()
    return getattr(layer, '__name__',
                   layer.__class__.__name__)