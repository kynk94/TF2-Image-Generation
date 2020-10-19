import tqdm
import tensorflow as tf

from utils import make_1d_latent, get_config, ImageLoader
from models import GAN, DCGAN

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def train_GAN():
    conf = get_config('configs/GAN/mnist.yaml')
    model = GAN(conf, use_log=True)

    loader = ImageLoader(data_txt_file=conf['train_data_txt'])
    train_dataset = loader.get_dataset(conf, flatten=True)
    test_data = make_1d_latent(batch=conf['test_batch_size'],
                               latent_dim=conf['latent_dim'],
                               seed=conf['random_seed'])

    pbar = tqdm.tqdm(range(1, conf['epochs']+1), position=0, leave=True)
    for epoch in pbar:
        for iteration, image_batch in enumerate(train_dataset):
            loss_g, loss_d = model.train(image_batch)

            if (iteration + 1) % 10 == 0:
                pbar.set_postfix({'G': '{:.4f}'.format(loss_g),
                                  'D': '{:.4f}'.format(loss_d)})
        if epoch % 1 == 0:
            model.test(test_data, epoch)


def train_DCGAN():
    conf = get_config('configs/DCGAN/cifar10.yaml')
    model = DCGAN(conf, use_log=True)

    loader = ImageLoader(data_txt_file=conf['train_data_txt'])
    train_dataset = loader.get_dataset(conf)
    test_data = make_1d_latent(batch=conf['test_batch_size'],
                               latent_dim=conf['latent_dim'],
                               seed=conf['random_seed'])

    pbar = tqdm.tqdm(range(1, conf['epochs']+1), position=0, leave=True)
    for epoch in pbar:
        for iteration, image_batch in enumerate(train_dataset):
            loss_g, loss_d = model.train(image_batch)

            if (iteration + 1) % 10 == 0:
                pbar.set_postfix({'G': '{:.4f}'.format(loss_g),
                                  'D': '{:.4f}'.format(loss_d)})
        if epoch % 1 == 0:
            model.test(test_data, epoch)
        if epoch % 5 == 0:
            model.save()


def main():
    train_DCGAN()


if __name__ == '__main__':
    main()
