import argparse
import tqdm
import tensorflow as tf

from utils import str_to_bool, get_config, find_config, check_dataset_config
from utils import allow_memory_growth, ImageLoader
from models import CGAN


def make_test_data(numeric_labels,
                   test_batch_size,
                   n_class,
                   latent_dim,
                   latent_each_class=False,
                   seed=None):
    div, mod = divmod(test_batch_size, n_class)
    if mod:
        raise ValueError("'testbatch_size' is not a multiple of 'n_class'.")

    label = tf.repeat(tf.Variable(numeric_labels, dtype=tf.float32), div)
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
                            default='configs/CGAN/mnist.yaml')
    arg_parser.add_argument('-ckpt', '--checkpoint', type=str,
                            default=None)
    args = vars(arg_parser.parse_args())

    if args['memory_growth']:
        allow_memory_growth()
    if args['checkpoint'] is not None:
        args['config'] = find_config(args['checkpoint'])

    conf = get_config(args['config'])
    check_dataset_config(conf)

    """Load Dataset"""
    loader = ImageLoader(data_txt_file=conf['train_data_txt'], use_label=True)
    conf['n_class'] = loader.n_class

    train_dataset = loader.get_dataset(batch_size=conf['batch_size'],
                                       flatten=True)

    labels = loader.get_label(str_label=map(str, range(conf['n_class'])))
    test_data = make_test_data(numeric_labels=labels,
                               test_batch_size=conf['test_batch_size'],
                               n_class=conf['n_class'],
                               latent_dim=conf['latent_dim'],
                               latent_each_class=True,
                               seed=conf['random_seed'])
    display_shape = (conf['n_class'],
                     conf['test_batch_size'] // conf['n_class'])

    """Model Initiate"""
    model = CGAN(conf, args['checkpoint'])
    if args['checkpoint'] is None:
        model.copy_conf(args['config'])

    """Start Train"""
    start_epoch = model.ckpt.step // len(train_dataset) + 1
    pbar = tqdm.trange(start_epoch, conf['epochs']+1,
                       position=0, leave=True)
    for epoch in pbar:
        for iteration, image_batch in enumerate(train_dataset):
            log_dict = model.train(image_batch)
            loss_g = log_dict['loss/gen']
            loss_d = log_dict['loss/dis']

            if (iteration + 1) % 10 == 0:
                pbar.set_postfix({'Current Epoch': epoch,
                                  'G': '{:.4f}'.format(loss_g),
                                  'D': '{:.4f}'.format(loss_d)})
        if epoch % 1 == 0:
            model.test(test_data, epoch, save=True,
                       display_shape=display_shape)


if __name__ == '__main__':
    main()
