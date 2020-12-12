"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import os
import glob
import argparse
import tqdm
import imageio
import numpy as np
from PIL import Image
from utils import NumericStringSort, float_0_to_1, extension_pattern

IMAGE_EXT = {'jpg', 'jpeg', 'png'}


def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('-i', '--input', type=str, required=True,
                           help='Input images directory')
    arg_parse.add_argument('-o', '--output', type=str,
                           default='output.gif',
                           help='Output file name')
    arg_parse.add_argument('-f', '--fps', type=float,
                           default=12,
                           help='Frames per Second')
    arg_parse.add_argument('-r', '--resolution', type=int,
                           default=None,
                           help='Output file resolution')
    frame_split = arg_parse.add_mutually_exclusive_group()
    frame_split.add_argument('-fc', '--frames_consecutive', type=int,
                             help='Total consecutive frames of gif counting from scratch')
    frame_split.add_argument('-fsr', '--frames_space_rate', type=float_0_to_1,
                             help='Rate of total frames from start to end (if 0.5, use half of frames)')
    frame_split.add_argument('-fi', '--frames_interval', type=int,
                             help='Interval index between adjacent frames (if 10, images=[0, 10, 20, ...])')
    args = vars(arg_parse.parse_args())

    if not args['output'].lower().endswith('.gif'):
        args['output'] += '.gif'

    images = []
    for EXT in IMAGE_EXT:
        images.extend(glob.glob(os.path.join(args['input'],
                                             extension_pattern(EXT)),
                                recursive=True))
    assert images, 'Image file not found'

    images.sort(key=NumericStringSort)
    print(f'Found {len(images)} images')

    if args['frames_consecutive'] is not None:
        images = images[:args['frames_consecutive']]
        print(f'Select {len(images)} images by frames_consecutive')
    elif args['frames_space_rate'] is not None:
        indexes = np.linspace(0, len(images)-1,
                              int(len(images)*args['frames_space_rate']),
                              dtype=np.int32)
        images = [images[i] for i in indexes]
        print(f'Select {len(images)} images by frames_space_rate')
    elif args['frames_interval'] is not None:
        fi = args['frames_interval']
        images = [images[i * fi] for i in range(len(images) // fi)]
        print(f'Select {len(images)} images by frames_interval')

    resolution = args['resolution']
    images_array = []
    pbar = tqdm.tqdm(images, position=0, leave=True)
    for image in pbar:
        image = Image.open(image)
        if resolution is not None:
            image = image.resize((resolution, resolution))
        images_array.append(image)

    print('GIF write start')

    try:
        imageio.mimsave(uri=args['output'],
                        ims=images_array,
                        format='GIF-FI',
                        fps=args['fps'],
                        quantizer='nq')
    except RuntimeError as e:
        print(e)
        print('Download FreeImage automatically')
        imageio.plugins.freeimage.download()
        imageio.mimsave(uri=args['output'],
                        ims=images_array,
                        format='GIF-FI',
                        fps=args['fps'],
                        quantizer='nq')
    finally:
        print('GIF write successful')


if __name__ == '__main__':
    main()
