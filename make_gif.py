import os
import glob
import argparse
import tqdm
import imageio

IMAGE_EXT = {'jpg', 'jpeg', 'png'}


def extension_pattern(extension):
    pattern = ''.join(f'[{e.lower()}{e.upper()}]' for e in extension)
    return f'*.{pattern}'


def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('-i', '--input', type=str,
                           help='Input images directory')
    arg_parse.add_argument('-o', '--output', type=str,
                           default='output.gif',
                           help='Output file name')
    arg_parse.add_argument('-f', '--fps', type=float,
                           default=20,
                           help='Frame per Second')
    args = vars(arg_parse.parse_args())

    if not args['output'].lower().endswith('.gif'):
        args['output'] += '.gif'

    writer = imageio.get_writer(args['output'],
                                mode='I',
                                duration=1/args['fps'])

    images = []
    for EXT in IMAGE_EXT:
        images.extend(glob.glob(os.path.join(args['input'],
                                             extension_pattern(EXT))))
    images.sort()

    for image in tqdm.tqdm(images):
        image = imageio.imread(image)
        writer.append_data(image)

    writer.close()


if __name__ == '__main__':
    main()
