import sys
import tqdm
import tensorflow as tf

from models import GAN
from utils import get_config

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():
    conf = get_config('config/mnist.yaml')
    model = GAN(conf, use_log=True)
    mnist = tf.keras.datasets.mnist.load_data()

    train_images = mnist[0][0] / 255 - 0.5
    train_images = train_images.reshape(train_images.shape[0], -1)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(60000).batch(conf['batch_size'])
    test_data = tf.random.normal(shape=(16, conf['latent_dim']),
                                 seed=conf['random_seed'])

    pbar = tqdm.tqdm(range(1, conf['epochs']+1), position=0, leave=True)
    for epoch in pbar:
        for iteration, image_batch in enumerate(train_dataset):
            loss_g, loss_d = model.train(image_batch)

            if (iteration + 1) % 50 == 0:
                pbar.set_postfix({'G': '{:.4f}'.format(loss_g),
                                  'D': '{:.4f}'.format(loss_d)})
        if epoch % 1 == 0:
            model.test(test_data, epoch)


if __name__ == '__main__':
    main()
