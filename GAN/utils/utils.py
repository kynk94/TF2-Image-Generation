import os
import glob
import yaml
import tensorflow as tf


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
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return


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
    data_conf = config['dataset']
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
        config['train_data_txt'], config['test_data_txt'] = txt_output
    else:
        config['train_data_txt'] = txt_output


def tf_image_concat(images, display_shape):
    n_row, n_col = display_shape
    output = []
    for i in range(n_row):
        output.append(tf.concat([*images[n_col*i:n_col*(i+1)]], axis=0))
    return tf.concat(output, axis=1)


def tf_image_write(filename, contents, denorm=True):
    if denorm:
        contents = contents * 127.5 + 127.5
    tf.io.write_file(filename=filename,
                     contents=tf.io.encode_png(tf.cast(contents, tf.uint8)))


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

    with open(output_path, 'w', encoding='utf-8') as txt_file:
        for root, _, files in os.walk(data_dir):
            if not files:
                continue
            if labeled_dir:
                label = ',' + os.path.basename(root)
            else:
                label = ''

            split_root = os.path.join(prefix, root[split_length:])
            for file_name in files:
                file_path = os.path.join(split_root, file_name)
                txt_file.write(f'{file_path}{label}\n')
    return output_path
