import os
import glob
import argparse
import re
import tqdm
import numpy as np
from PIL import Image
from utils import NumericStringSort, extension_pattern, str_to_bool

IMAGE_EXT = {'jpg', 'jpeg', 'png'}


def string_indexing_array_2D(arr, r, c, assign):
    r_count, c_count = 0, 0
    r_split, c_split = [None] * 3, [None] * 3
    for i, split in enumerate(re.split('(:)', r)):
        if split == ':':
            r_count += 1
            continue
        if split:
            r_split[i-r_count] = int(split)
    for i, split in enumerate(re.split('(:)', c)):
        if split == ':':
            c_count += 1
            continue
        if split:
            c_split[i-c_count] = int(split)
    if r_count == 0:
        new = arr[r_split[0]]
    elif r_count == 1:
        new = arr[r_split[0]:r_split[1]]
    else:
        new = arr[r_split[0]:r_split[1]:r_split[2]]
    if c_count == 0:
        new[..., c_split[0]] = assign
    elif c_count == 1:
        new[..., c_split[0]:c_split[1]] = assign
    else:
        new[..., c_split[0]:c_split[1]:c_split[2]] = assign


def make_patch_info(n_row, n_col):
    def input_to_mode():
        while True:
            value = input('Select mode {add, remove}: ')
            v_lower = value.lower()
            if v_lower in {'add', 'a'}:
                return 'add'
            if v_lower in {'remove', 'r'}:
                return 'remove'
            print(f'Wrong mode {value}')

    mode = input_to_mode()
    assign = mode == 'add'
    patch_info = np.ones((n_row, n_col), dtype=np.int32) - assign

    input_string = ('(input "add" or "remove" to change mode, input "make" to make output, ' +
                    'input "length" to get number of selected patches)\n' +
                    'Enter the indexing used by numpy: ')
    while True:
        print('\nCurrent Patches:')
        print(patch_info, end='\n\n')
        index_input = input(input_string).lower()
        if index_input in {'length', 'len', 'l'}:
            print(f'Number of selected patches: {np.sum(patch_info)}')
            continue
        if index_input in {'add', 'remove', 'a', 'r'}:
            assign = 'a' in index_input
            continue
        if index_input == 'make':
            return patch_info

        indexes = re.sub(r'[^-:\d]+', ' ', index_input)
        index_split = indexes.split()
        if '--' in indexes or len(index_split) > 2:
            print(f'Wrong indexing input: {index_input}')
            continue

        indexes = [':'] * 2
        for i, c in enumerate(index_split):
            indexes[i] = c

        string_indexing_array_2D(patch_info, *indexes, assign)


def maximum_two_divisor(num):
    for i in range(int(num**0.5), 0, -1):
        div, mod = divmod(num, i)
        if mod == 0:
            return sorted([div, i])
    return 1, num


def mod_patch_info(patch_info, auto_square, n_target=None):
    grid_index = np.where(patch_info.reshape(-1) == 1)
    n_patch = len(grid_index[0])
    out_n_row, out_n_col = maximum_two_divisor(n_patch)

    print(f'Number of selected patch: {n_patch}')
    print(f'Output Shape = {out_n_row} x {out_n_col}')

    if n_target is None and (
            not auto_square or abs(out_n_row - out_n_col) < 2):
        return out_n_row, out_n_col

    out_n_patch = n_target if n_target else round(n_patch**0.5)**2
    out_n_row, out_n_col = maximum_two_divisor(out_n_patch)
    n_need_select = out_n_patch - n_patch

    compare = n_need_select < 0
    points = list(zip(*np.where(patch_info == compare)))
    choices = np.random.choice(len(points), abs(n_need_select), replace=False)

    points = [points[choice] for choice in choices]
    patch_info[tuple(zip(*points))] = not compare

    print('\nAuto Squared Patches:')
    print(patch_info, end='\n\n')
    print(f'Number of auto squared patches: {out_n_patch}')
    print(f'Auto Squared Output Shape = {out_n_row} x {out_n_col}')
    return out_n_row, out_n_col


def extract_patches(arr, n_row, n_col):
    width, height, channel = arr.shape
    patch_shape = (width // n_col, height // n_row, channel)
    n_patch_axis = (n_col, n_row, 1)

    shape = n_patch_axis + patch_shape
    strides = tuple(np.array(arr.strides) * patch_shape) + arr.strides

    patches = np.lib.stride_tricks.as_strided(arr, shape, strides)
    return patches.reshape(-1, *patch_shape)


def merge_patches(patches, n_row, n_col):
    n_patches, width, height, channel = patches.shape
    assert n_patches == n_row * n_col, 'Shape does not match'

    merged = np.zeros((height * n_row, width * n_col, channel),
                      dtype=np.uint8)
    patch_iter = iter(patches)
    for r in range(n_row):
        for c in range(n_col):
            merged[r*width:(r+1)*width,
                   c*height:(c+1)*height] = next(patch_iter)
    return merged


def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('-i', '--input', type=str, required=True,
                           help='Input images directory')
    arg_parse.add_argument('-o', '--output', type=str,
                           default='./output/grids',
                           help='Output directory name (default=./output/grids')
    arg_parse.add_argument('-n', '--n_target', type=int,
                           default=None,
                           help='Target number of patches in output')
    arg_parse.add_argument('-r', '--row', type=int, required=True,
                           help='Number of rows in input images')
    arg_parse.add_argument('-c', '--col', type=int, required=True,
                           help='Number of columns in input images')
    arg_parse.add_argument('-as', '--auto_square', type=str_to_bool,
                           default=True,
                           help='Flag. Make Selected Patches to almost square')
    args = vars(arg_parse.parse_args())

    n_row, n_col = args['row'], args['col']

    images = []
    for EXT in IMAGE_EXT:
        images.extend(glob.glob(os.path.join(args['input'],
                                             extension_pattern(EXT)),
                                recursive=True))
    assert images, 'Image file not found'

    images.sort(key=NumericStringSort)
    print(f'Found {len(images)} images')

    patch_info = make_patch_info(n_row, n_col)
    out_n_row, out_n_col = mod_patch_info(patch_info,
                                          args['auto_square'],
                                          args['n_target'])
    patch_index = np.where(patch_info.reshape(-1) == 1)[0]

    basename = os.path.basename(os.path.dirname(images[0]))
    out_dir = os.path.join(args['output'], basename)
    os.makedirs(out_dir, exist_ok=True)
    for image in tqdm.tqdm(images):
        out_path = os.path.join(out_dir, os.path.basename(image))
        arr = np.array(Image.open(image))
        patches = extract_patches(arr, n_row, n_col)
        selected_patches = patches[patch_index]
        merged = merge_patches(selected_patches, out_n_row, out_n_col)
        Image.fromarray(merged).save(out_path)


if __name__ == '__main__':
    main()
