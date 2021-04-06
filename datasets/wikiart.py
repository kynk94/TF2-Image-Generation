import os
import glob
import tqdm
import argparse
import numpy as np
import multiprocessing
from PIL import Image


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def run(inputs):
    index, images, args = inputs
    resolution = args['resolution']
    output_dir = args['output_dir']
    pbar = tqdm.tqdm(images, position=index, leave=True)
    for image in pbar:
        genre = os.path.basename(os.path.dirname(image))
        output_path = os.path.join(output_dir, genre)
        image_name = os.path.basename(image)
        os.makedirs(output_path, exist_ok=True)

        i = Image.open(image)
        if not np.array(i).shape:
            pbar.set_postfix({
                'f': image_name,
                'e': 'load_fail',
                'size': (w, h)})
            continue

        w, h = i.size
        if min(w, h) < resolution:
            i.save(os.path.join(output_path, image_name))
            continue

        pbar.set_postfix({
            'f': image_name,
            'e': 'resize',
            'size': (w, h)})
        scale = min(w, h) / resolution
        i.resize((int(w // scale), int(h // scale))).save(
            os.path.join(output_path, image_name))


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-p', '--path', default='wikiart',
                            help='Path of unzipped wikiart directory', type=str)
    arg_parser.add_argument('-o', '--output_dir', default='resized_wikiart',
                            help='Path of output directory', type=str)
    arg_parser.add_argument('-r', '--resolution', default=1024,
                            help='Target Resolution', type=int)
    arg_parser.add_argument('-n', '--n_process', default=8,
                            help='Number of core to use', type=int)
    args = vars(arg_parser.parse_args())

    images = glob.glob(os.path.join(args['path'], '**/*.*g'), recursive=True)
    assert images, 'Image files not found.'

    divided_list = list(split(images, args['n_process']))
    inputs = [(i, divided, args) for i, divided in enumerate(divided_list)]
    pool = multiprocessing.Pool(processes=args['n_process'])
    pool.map(run, inputs)
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
