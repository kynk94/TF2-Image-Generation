import tqdm
import argparse
import tensorflow as tf

from utils import allow_memory_growth, get_config, check_dataset_config
from utils import ImageLoader
from models import GAN


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-mg', '--memory_growth',
                            default=True)
    arg_parser.add_argument('-c', '--config',
                            default='configs/GAN/mnist.yaml')
    args = vars(arg_parser.parse_args())

    if args['memory_growth']:
        allow_memory_growth()

    conf_path = args['config']
    conf = get_config(conf_path)
    check_dataset_config(conf)

    model = GAN(conf)
    model.copy_conf(conf_path)

    loader = ImageLoader(data_txt_file=conf['train_data_txt'])
    train_dataset = loader.get_dataset(conf, flatten=True)
    test_data = tf.random.normal(shape=(conf['test_batch_size'], conf['latent_dim']),
                                 seed=conf['random_seed'])

    step = 0  # step = model.load(ckpt)
    start_epoch = step // len(train_dataset) + 1
    pbar = tqdm.trange(start_epoch, conf['epochs']+1,
                       position=0, leave=True)
    for epoch in pbar:
        for iteration, image_batch in enumerate(train_dataset):
            loss_g, loss_d = model.train(image_batch)

            if (iteration + 1) % 10 == 0:
                pbar.set_postfix({'Current Epoch': epoch,
                                  'G': '{:.4f}'.format(loss_g),
                                  'D': '{:.4f}'.format(loss_d)})
        if epoch % 1 == 0:
            model.test(test_data, epoch)


if __name__ == '__main__':
    allow_memory_growth()
    main()
