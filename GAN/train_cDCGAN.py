import math
import argparse
import tqdm
import tensorflow as tf

from utils import str_to_bool, get_config, find_config, check_dataset_config
from utils import allow_memory_growth, ImageLoader
from models import ConditionalDCGAN


def make_test_data(numeric_labels,
                   test_batch_size,
                   n_class,
                   latent_dim,
                   latent_each_class=False,
                   seed=None):
    div, mod = divmod(test_batch_size, n_class)
    if mod:
        raise ValueError("'testbatch_size' is not a multiple of 'n_class'.")

    label = tf.repeat(tf.Variable(numeric_labels, dtype=tf.int32), div)
    label = tf.one_hot(label, n_class, dtype=tf.float32)
    if latent_each_class:
        latent = tf.random.normal(shape=(div, latent_dim),
                                  seed=seed)
        latent = tf.tile(latent, (n_class, 1))
    else:
        latent = tf.random.normal(shape=(test_batch_size, latent_dim))

    return latent, label


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-mg', '--memory_growth', type=str_to_bool,
                            default=True)
    arg_parser.add_argument('-c', '--config', type=str,
                            default='configs/cDCGAN/cifar10.yaml')
    arg_parser.add_argument('-ckpt', '--checkpoint', type=str,
                            default=None)
    args = vars(arg_parser.parse_args())

    tf.keras.backend.set_image_data_format('channels_first')

    if args['memory_growth']:
        allow_memory_growth()
    if args['checkpoint'] is not None:
        args['config'] = find_config(args['checkpoint'])

    conf = get_config(args['config'])
    check_dataset_config(conf)

    """Load Dataset"""
    dataset_conf = conf['dataset']
    loader = ImageLoader(data_txt_file=dataset_conf['train_data_txt'],
                         use_label=True)
    conf['n_class'] = loader.n_class

    def map_func(image, label):
        label = tf.one_hot(tf.cast(label, tf.int32), conf['n_class'],
                           dtype=tf.float32)
        return image, label

    train_dataset = loader.get_dataset(batch_size=conf['batch_size'],
                                       map_func=map_func,
                                       new_size=(conf['input_size'],)*2,
                                       cache=dataset_conf['cache'])

    labels = loader.get_label(str_label=sorted(loader.class_dict))
    test_data = make_test_data(numeric_labels=labels,
                               test_batch_size=conf['test_batch_size'],
                               n_class=conf['n_class'],
                               latent_dim=conf['latent_dim'],
                               latent_each_class=True,
                               seed=conf['random_seed'])
    display_shape = (conf['n_class'],
                     conf['test_batch_size'] // conf['n_class'])

    """Model Initiate"""
    strategy = tf.distribute.MirroredStrategy()
    n_replica = strategy.num_replicas_in_sync
    model = ConditionalDCGAN(conf, args['checkpoint'], strategy)
    if args['checkpoint'] is None:
        model.copy_conf(args['config'])

    """Start Train"""
    start_epoch = model.ckpt.step // len(train_dataset) + 1
    epoch_by_step = math.ceil(conf['steps'] / len(train_dataset))
    if conf['epochs'] < epoch_by_step:
        conf['epochs'] = epoch_by_step
    else:
        conf['steps'] = conf['epochs'] * len(train_dataset)
    test_step = conf['test_step']
    save_step = conf['save_step']
    end_step = conf['steps']

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    len_dist_dataset = len(train_dataset) // n_replica
    pbar = tqdm.trange(start_epoch, conf['epochs']+1,
                       position=0, leave=True)
    pbar_dict = dict()
    for epoch in pbar:
        pbar.set_postfix({'Current Epoch': epoch})
        sub_pbar = tqdm.tqdm(train_dist_dataset, total=len_dist_dataset,
                             leave=False)
        for image_batch in sub_pbar:
            log_dict = model.train(image_batch)

            current_step = model.ckpt.step.numpy()
            if current_step % 10 == 0:
                pbar_dict.update({
                    'Step': current_step,
                    'G': '{:.4f}'.format(log_dict['loss/gen']),
                    'D': '{:.4f}'.format(log_dict['loss/dis'])})
                sub_pbar.set_postfix(pbar_dict)
            if test_step and current_step % test_step == 0:
                model.test(test_data, current_step, save=True,
                           display_shape=display_shape)
            if save_step and current_step % save_step == 0:
                model.save()
            if current_step == end_step:
                return


if __name__ == '__main__':
    main()
