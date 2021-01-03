import argparse
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
                            default='content')
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
    content_batch_size, _, *input_wh = content_image.shape
    style_image, _ = read_images(args['input_style'], shape=input_wh)
    style_batch_size = style_image.shape[0]

    if style_batch_size != 1:
        if content_batch_size == 1:
            content_image = tf.repeat(content_image, style_batch_size, axis=0)
            init_shape = tf.repeat(init_shape, style_batch_size, axis=0)
        elif content_batch_size != style_batch_size:
            raise ValueError('When using multiple content images and multiple style images,' +
                             'the number of images should be equal.')

    init_image = None
    if args['mode'] == 'latent':
        init_image = tf.random.normal(content_image.shape)
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
            pbar_dict.update({
                'Step': current_step,
                'Content': '{:.4f}'.format(log_dict['loss/content']),
                'Style': '{:.4f}'.format(log_dict['loss/style'])
            })
            pbar.set_postfix(pbar_dict)
        if test_step and current_step % test_step == 0:
            model.write_drawing_image(init_shape, current_step, save=True)


if __name__ == '__main__':
    main()
