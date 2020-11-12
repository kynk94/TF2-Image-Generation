import argparse
from numpy.lib.arraysetops import isin
import tqdm
import tensorflow as tf

from utils import str_to_bool, get_config
from utils import allow_memory_growth, read_images
from models import NeuralStyleTransfer


def str_to_mode(value):
    v_lower = value.lower()
    if v_lower in {'latent', 'l', 'random', 'r'}:
        return 'latent'
    elif v_lower in {'content', 'c'}:
        return 'content'
    elif v_lower in {'style', 's'}:
        return 'style'
    raise ValueError(f'{value} is not a valid mode value')


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-mg', '--memory_growth', type=str_to_bool,
                            default=True)
    arg_parser.add_argument('-c', '--config', type=str,
                            default='configs/NST/vgg19.yaml')
    arg_parser.add_argument('-m', '--mode', type=str_to_mode,
                            default='latent')
    arg_parser.add_argument('-ic', '--input_content', type=str,
                            default='../dataset/style_transfer/content')
    arg_parser.add_argument('-is', '--input_style', type=str,
                            default='../dataset/style_transfer/style')
    args = vars(arg_parser.parse_args())

    tf.keras.backend.set_image_data_format('channels_first')

    if args['memory_growth']:
        allow_memory_growth()

    conf = get_config(args['config'])

    """Load Dataset"""
    content_image, init_shape = read_images(args['input_content'])
    content_image = content_image / 2 + 0.5
    input_wh = content_image.shape[2:]
    style_image, _ = read_images(args['input_style'], shape=input_wh)
    style_image = style_image / 2 + 0.5
    init_image = None
    if args['mode'] == 'latent':
        init_image = tf.random.normal(content_image.shape, mean=0.5)
    elif args['mode'] == 'content':
        init_image = content_image
    elif args['mode'] == 'style':
        init_image = style_image

    """Model Initiate"""
    model = NeuralStyleTransfer(conf, init_image, content_image, style_image)

    """Start Train"""
    test_step = conf['test_step']
    pbar = tqdm.trange(conf['steps'], position=0, leave=True)
    pbar_dict = dict()
    for _ in pbar:
        log_dict = model.train()

        current_step = model.ckpt.step.numpy()
        if current_step % 10 == 0:
            pbar_dict['Step'] = current_step
            pbar_dict['C'] = '{:.4f}'.format(log_dict['loss/content'])
            pbar_dict['S'] = '{:.4f}'.format(log_dict['loss/style'])
            pbar.set_postfix(pbar_dict)
        if test_step and current_step % test_step == 0:
            model.write_init_image(init_shape, current_step, save=True)


if __name__ == '__main__':
    main()
