import tqdm

from utils import allow_memory_growth, make_1d_latent, get_config, ImageLoader
from models import GAN


def main():
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
                pbar.set_postfix({'Current Epoch': epoch,
                                  'G': '{:.4f}'.format(loss_g),
                                  'D': '{:.4f}'.format(loss_d)})
        if epoch % 1 == 0:
            model.test(test_data, epoch)


if __name__ == '__main__':
    allow_memory_growth()
    main()
