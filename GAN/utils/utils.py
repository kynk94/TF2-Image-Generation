import os
import yaml
import tensorflow as tf


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def make_1d_latent(batch, latent_dim, seed=None):
    return tf.random.normal(shape=(batch, latent_dim), seed=seed)


def tf_image_concat(images, display_shape):
    n_row, n_col = display_shape
    output = []
    for i in range(n_col):
        output.append(tf.concat([*images[n_row*i:n_row*(i+1)]], axis=0))
    return tf.concat(output, axis=1)


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
