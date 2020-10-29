import tqdm

from utils import allow_memory_growth, make_1d_latent, get_config, check_dataset_config
from utils import ImageLoader
from models import DCGAN


def main():
    conf = get_config('configs/DCGAN/cifar10.yaml')
    check_dataset_config(conf)

    model = DCGAN(conf, use_log=True)

    loader = ImageLoader(data_txt_file=conf['train_data_txt'])
    train_dataset = loader.get_dataset(batch_size=conf['batch_size'],
                                       new_size=(conf['input_size'],)*2)
    test_data = make_1d_latent(batch=conf['test_batch_size'],
                               latent_dim=conf['latent_dim'],
                               seed=conf['random_seed'])

    step = 0  # step = model.load(ckpt)
    start_epoch = step // len(train_dataset) + 1
    pbar = tqdm.tqdm(range(start_epoch, conf['epochs']+1),
                     position=0, leave=True)
    for epoch in pbar:
        for iteration, image_batch in enumerate(train_dataset):
            loss_g, loss_d = model.train(image_batch)

            if (iteration + 1) % 10 == 0:
                pbar.set_postfix({'Current Epoch': epoch,
                                  'G': '{:.4f}'.format(loss_g),
                                  'D': '{:.4f}'.format(loss_d)})
        if epoch % 1 == 0:
            model.test(test_data, epoch, save=True)
        if epoch % 5 == 0:
            model.save()


if __name__ == '__main__':
    allow_memory_growth()
    main()
