import math
import argparse
import tqdm
import tensorflow as tf

from utils import str_to_bool, get_config, find_config, check_dataset_config
from utils import allow_memory_growth, ImageLoader
from models import AdaIN


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-mg', '--memory_growth', type=str_to_bool,
                            default=True)
    arg_parser.add_argument('-c', '--config', type=str,
                            default='configs/AdaIN/coco14_wikiart.yaml')
    arg_parser.add_argument('-ckpt', '--checkpoint', type=str,
                            default=None)
    args = vars(arg_parser.parse_args())

    tf.keras.backend.set_image_data_format('channels_first')

    if args['memory_growth']:
        allow_memory_growth()
    if args['checkpoint'] is not None:
        args['config'] = find_config(args['checkpoint'])

    conf = get_config(args['config'])
    check_dataset_config(conf['dataset']['content'])
    check_dataset_config(conf['dataset']['style'])

    """Load Dataset"""
    dataset_conf = conf['dataset']
    content_conf = dataset_conf['content']
    style_conf = dataset_conf['style']
    content_loader = ImageLoader(data_txt_file=content_conf['train_data_txt'])
    content_dataset = content_loader.get_dataset(batch_size=conf['batch_size'],
                                               new_size=(
                                                   conf['input_size'],)*2,
                                               cache=content_conf['cache'])
    test_content = next(iter(
        content_loader.get_dataset(batch_size=conf['test_batch_size'],
                                   new_size=(conf['input_size'],)*2)))
    style_loader = ImageLoader(data_txt_file=style_conf['train_data_txt'])
    style_dataset = style_loader.get_dataset(batch_size=conf['batch_size'],
                                             new_size=(conf['input_size'],)*2,
                                             cache=style_conf['cache'])
    train_dataset = tf.data.Dataset.zip((content_dataset, style_dataset)).repeat()
    test_style = next(iter(
        style_loader.get_dataset(batch_size=conf['test_batch_size'],
                                 new_size=(conf['input_size'],)*2)))
    test_data = (test_content, test_style)

    """Model Initiate"""
    model = AdaIN(conf, args['checkpoint'])
    if args['checkpoint'] is None:
        model.copy_conf(args['config'])
        model.test(test_data, save_input=True)

    """Start Train"""
    start_step = model.ckpt.step.numpy()
    test_step = conf['test_step']
    save_step = conf['save_step']
    end_step = conf['steps']

    pbar = tqdm.trange(start_step, end_step,
                       position=0, leave=True)
    pbar_dict = dict()
    train_iter = iter(train_dataset)
    for _ in pbar:
        image_batch = next(train_iter)
        log_dict = model.train(image_batch)

        current_step = model.ckpt.step.numpy()

        if current_step % 1 == 0:
            pbar_dict.update({
                'Step': current_step,
                'Content': '{:.4f}'.format(log_dict['loss/content']),
                'Style': '{:.4f}'.format(log_dict['loss/style'])
            })
            pbar.set_postfix(pbar_dict)
        if test_step and current_step % test_step == 0:
            model.test(test_data, current_step, save=True)
        if save_step and current_step % save_step == 0:
            model.save()
        if current_step == end_step:
            return


if __name__ == '__main__':
    main()
