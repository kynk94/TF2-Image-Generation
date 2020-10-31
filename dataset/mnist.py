import os
import glob
import shutil
import argparse
import gzip
import numpy as np
from collections import defaultdict
from tqdm import trange
from PIL import Image
from dataset_utils import check_archive


def read_mnist(image_file, label_file):
    with open(image_file, 'rb') as image_io, open(label_file, 'rb') as label_io:
        image_io.read(4)
        label_io.read(4)
        num_images = int.from_bytes(image_io.read(4), 'big')
        num_labels = int.from_bytes(label_io.read(4), 'big')
        assert num_images == num_labels, 'images and labels do not match.'

        height = int.from_bytes(image_io.read(4), 'big')
        width = int.from_bytes(image_io.read(4), 'big')
        bytes_to_read = height * width
        for i in trange(num_images):
            image = image_io.read(bytes_to_read)
            image = np.array([i for i in image], dtype=np.uint8)
            image = image.reshape(height, width)
            label = int.from_bytes(label_io.read(1), 'big')
            yield image, label


def write_images(image_file, label_file, prefix):
    index = defaultdict(lambda: 0)
    for image, label in read_mnist(image_file, label_file):
        image_dir = os.path.join(prefix, str(label))
        os.makedirs(image_dir, exist_ok=True)

        image_name = '{:05d}.png'.format(index[label])
        image_path = os.path.join(image_dir, image_name)
        Image.fromarray(image).save(image_path)
        index[label] += 1


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-a', '--archive_dir',
                            help='Directory of archive files', type=str)
    arg_parser.add_argument('-o', '--output_dir', default='./mnist',
                            help='Path of output directory', type=str)
    args = vars(arg_parser.parse_args())

    """Extract Archives"""
    archives = glob.glob(os.path.join(args['archive_dir'], '*ubyte.gz'))
    extract_path = os.path.join(args['output_dir'], 'archives')
    os.makedirs(extract_path, exist_ok=True)
    for archive in archives:
        archive_path = os.path.join(extract_path, check_archive(archive))
        with gzip.open(archive, 'rb') as f_in, open(archive_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    """Write Train/Test Images"""
    files = glob.glob(os.path.join(extract_path, '*'))
    test_images, test_labels, train_images, train_labels = sorted(files)

    write_images(train_images, train_labels,
                 os.path.join(args['output_dir'], 'train'))
    write_images(test_images, test_labels,
                 os.path.join(args['output_dir'], 'test'))

    shutil.rmtree(extract_path)


if __name__ == '__main__':
    main()
