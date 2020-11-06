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

    images = []
    for EXT in IMAGE_EXT:
        images.extend(glob.glob(os.path.join(args['input'],
                                             extension_pattern(EXT))))
    assert images, 'Image file not found'
    
    images.sort()
    print(f'Found {len(images)} images')

    images_array = []
    pbar = tqdm.tqdm(images, position=0, leave=True)
    for image in pbar:
        images_array.append(imageio.imread(image))

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
