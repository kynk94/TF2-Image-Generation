import os
import glob
import shutil
import argparse
import pickle
from tqdm import tqdm
from PIL import Image
from dataset_utils import check_archive


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-p', '--path',
                            help='Path of archive file', type=str)
    arg_parser.add_argument('-o', '--output_dir', default='./cifar10',
                            help='Path of output directory', type=str)
    args = vars(arg_parser.parse_args())

    """Extract Archive"""
    archive_name = check_archive(args['path'])
    extract_path = os.path.join(args['output_dir'], archive_name)
    shutil.unpack_archive(args['path'], extract_path)

    """Write Train/Test Images"""
    files = glob.glob(os.path.join(extract_path, '*/*batch*'))
    label, *batches = sorted(files)
    labels = [l.decode() for l in unpickle(label)[b'label_names']]

    train_test = ['train']*(len(batches)-1) + ['test']
    for t, batch in tqdm(tuple(zip(train_test, batches))):
        for label, data, image_name in zip(*list(unpickle(batch).values())[1:]):
            image_dir = os.path.join(args['output_dir'], t, labels[label])
            os.makedirs(image_dir, exist_ok=True)
            image = data.reshape(3, 32, 32).transpose(1, 2, 0)
            image_path = os.path.join(image_dir, image_name.decode())
            Image.fromarray(image).save(image_path)

    shutil.rmtree(extract_path)


if __name__ == '__main__':
    main()
