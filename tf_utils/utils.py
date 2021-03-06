import os
import glob
import yaml
import numpy as np
import tensorflow as tf
from PIL import Image

IMAGE_EXT = {'jpg', 'jpeg', 'png'}


def extension_pattern(extension):
    pattern = ''.join(f'[{e.lower()}{e.upper()}]' for e in extension)
    return f'**/*.{pattern}'


def find_images(path):
    if os.path.isfile(path):
        return [path]
    images = []
    for EXT in IMAGE_EXT:
        images.extend(glob.glob(os.path.join(path, extension_pattern(EXT)),
                                recursive=True))
    images.sort()
    assert images, 'Image file not found'
    return images


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    if value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def tf_image_concat(images, display_shape, mode='nchw'):
    n_row, n_col = display_shape
    if mode.lower() == 'nchw':
        images = tf.transpose(images, perm=(0, 2, 3, 1))
    output = []
    images = images[:n_row * n_col]
    output = tf.concat(
        tf.split(tf.reshape(images, (1, -1, *images.shape[2:])),
                 n_col, axis=1),
        axis=2)
    return output[0]


def tf_image_write(filename, contents, denorm=True):
    if denorm:
        contents = contents * 127.5 + 127.5
    tf.io.write_file(filename=filename,
                     contents=tf.io.encode_png(tf.cast(contents, tf.uint8)))


def tensor2image(tensor, denorm=True, mode='nchw'):
    if mode.lower() == 'nchw':
        tensor = tf.transpose(tensor, perm=(0, 2, 3, 1))
    if denorm:
        tensor = tf.clip_by_value(tensor, -1., 1.) * 127.5 + 127.5
    image = tf.cast(tensor, tf.uint8).numpy()
    return image


def image2tensor(image, normalize=True, mode='nchw', dtype=tf.float32):
    if isinstance(image, str):
        image = np.array(Image.open(image).convert('RGB'))
    tensor = tf.convert_to_tensor(image, dtype=dtype)
    if tensor.ndim == 3:
        tensor = tf.expand_dims(tensor, 0)
    if mode.lower() == 'nchw':
        tensor = tf.transpose(tensor, perm=(0, 3, 1, 2))
    if normalize:
        tensor = tensor / 127.5 - 1.0
    return tensor


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def find_config(checkpoint):
    if not os.path.isdir(checkpoint):
        checkpoint = os.path.dirname(checkpoint)
    config = glob.glob(os.path.join(checkpoint, '*.yaml'))
    assert config, 'Could not find config yaml file.'
    return config[0]


def check_dataset_config(config, make_txt=False):
    if 'dataset' in config:
        data_conf = config['dataset']
    else:
        data_conf = config
    if not make_txt and data_conf['train_data_txt'] is not None:
        if not os.path.exists(data_conf['train_data_txt']):
            raise FileNotFoundError("'train_data_txt' not found")
        return

    if not os.path.exists(data_conf['data_dir']):
        raise FileNotFoundError("'data_dir' not found")

    txt_output = make_dataset_txt(data_conf['data_dir'],
                                  train_test_split=data_conf['train_test_split'],
                                  labeled_dir=data_conf['labeled_dir'])

    if isinstance(txt_output, tuple):
        data_conf['train_data_txt'], data_conf['test_data_txt'] = txt_output
    else:
        data_conf['train_data_txt'] = txt_output


def make_dataset_txt(data_dir,
                     output_path=None,
                     train_test_split=False,
                     labeled_dir=False,
                     prefix=''):
    if train_test_split:
        if set(os.listdir(data_dir)).intersection(
                {'train', 'test'}) != {'train', 'test'}:
            raise FileNotFoundError(
                'The train or test directory is not exists in data_dir.')
        return (make_dataset_txt(data_dir=os.path.join(data_dir, 'train'),
                                 output_path=os.path.join(data_dir,
                                                          'train.txt'),
                                 train_test_split=False,
                                 labeled_dir=labeled_dir,
                                 prefix='train'),
                make_dataset_txt(data_dir=os.path.join(data_dir, 'test'),
                                 output_path=os.path.join(data_dir,
                                                          'test.txt'),
                                 train_test_split=False,
                                 labeled_dir=labeled_dir,
                                 prefix='test'))

    if output_path is None:
        output_path = os.path.join(data_dir, 'train.txt')
    if data_dir.endswith('/'):
        split_length = len(data_dir)
    else:
        split_length = len(data_dir) + 1

    IMAGE_EXT = {'jpg', 'jpeg', 'png'}

    def extension_pattern(extension):
        pattern = ''.join(f'[{e.lower()}{e.upper()}]' for e in extension)
        return f'**/*.{pattern}'

    images = []
    for EXT in IMAGE_EXT:
        images.extend(glob.glob(os.path.join(data_dir, extension_pattern(EXT)),
                                recursive=True))
    images.sort()

    with open(output_path, 'w', encoding='utf-8') as txt_file:
        for image in images:
            if labeled_dir:
                label = ',' + os.path.basename(os.path.dirname(image))
            else:
                label = ''
            line = f'{os.path.join(prefix, image[split_length:])}{label}\n'
            txt_file.write(line)
    return output_path
