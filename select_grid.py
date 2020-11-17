from os import replace
import re
import os
import glob
import argparse
import tqdm
import numpy as np
from PIL import Image
from utils import NumericStringSort, extension_pattern, str_to_bool

IMAGE_EXT = {'jpg', 'jpeg', 'png'}


def input_to_mode():
    while True:
        value = input('Select mode {add, remove}: ')
        v_lower = value.lower()
        if v_lower in {'add', 'a'}:
            return 'add'
        if v_lower in {'remove', 'r'}:
            return 'remove'
        print(f'Wrong mode {value}')


def array_assign(arr, r, c, assign):
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


def make_grid(image_grid=None, nrow=None, ncol=None):
    mode = input_to_mode()
    assign = mode == 'add'
    if image_grid is None:
        if nrow is None or ncol is None:
            raise ValueError('nrow or ncol should not be None')
        image_grid = np.ones((nrow, ncol), dtype=np.int32) - assign

    input_string = ('(input "mode" to change mode, input "make" to make output, ' +
                    'input "length" to get number of selected grid)\n' +
                    'Enter the indexing used by numpy: ')
    while True:
        print('\nCurrent Grid:')
        print(image_grid, end='\n\n')
        coordinate = input(input_string).lower()
        if coordinate in {'length', 'len', 'l'}:
            print(f'Current Selected: {np.sum(image_grid)}')
            continue
        if coordinate in {'mode', 'make'}:
            return coordinate, image_grid

        coordinates = re.sub(r'[^-:\d]+', ' ', coordinate)
        coord_split = coordinates.split()
        if '--' in coordinates or len(coord_split) > 2:
            print(f'Wrong coordinate: {coordinate}')
            continue

        coordinates = [':'] * 2
        for i, c in enumerate(coord_split):
            coordinates[i] = c

        array_assign(image_grid, *coordinates, assign)


def mod_selected_grid(image_grid, auto_square):
    grid_index = np.where(image_grid.reshape(-1) == 1)
    n_grid = len(grid_index[0])
    out_nrow, out_ncol = 0, 0
    for i in range(int(n_grid**0.5), 0, -1):
        div, mod = divmod(n_grid, i)
        if mod == 0:
            out_nrow, out_ncol = sorted([div, i])
            break

    print(f'Number of selected grid: {n_grid}')
    print(f'Output Shape = {out_nrow} x {out_ncol}')

    if not auto_square or abs(out_nrow - out_ncol) < 2:
        return out_nrow, out_ncol

    out_n_grid = round(n_grid**0.5)**2
    out_nrow = out_ncol = int(out_n_grid**0.5)
    n_need_select = out_n_grid - n_grid

    compare = n_need_select < 0
    points = list(zip(*np.where(image_grid == compare)))
    choices = np.random.choice(len(points), abs(n_need_select), replace=False)

    points = [points[choice] for choice in choices]
    image_grid[tuple(zip(*points))] = not compare

    print('\nAuto Squared Grid:')
    print(image_grid, end='\n\n')
    print(f'Number of auto squared grid: {out_n_grid}')
    print(f'Auto Squared Output Shape = {out_nrow} x {out_ncol}')
    return out_nrow, out_ncol


def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('-i', '--input', type=str, required=True,
                           help='Input images directory')
    arg_parse.add_argument('-o', '--output', type=str,
                           default='./output/grids',
                           help='Output directory name (default=./output/grids')
    arg_parse.add_argument('-nr', '--nrow', type=int,
                           help='Number of rows in input images')
    arg_parse.add_argument('-nc', '--ncol', type=int,
                           help='Number of columns in input images')
    arg_parse.add_argument('-as', '--auto_square', type=str_to_bool,
                           default=True,
                           help='Flag. Make Selected Grid to almost square')
    args = vars(arg_parse.parse_args())

    images = []
    for EXT in IMAGE_EXT:
        images.extend(glob.glob(os.path.join(args['input'],
                                             extension_pattern(EXT)),
                                recursive=True))
    assert images, 'Image file not found'
    width, height = Image.open(images[0]).size
    nrow, ncol = args['nrow'], args['ncol']
    grid_width = width // ncol
    grid_height = height // nrow

    images.sort(key=NumericStringSort)
    print(f'Found {len(images)} images')

    image_grid = None
    while True:
        mode, image_grid = make_grid(image_grid, nrow, ncol)
        if mode == 'make':
            break
    out_nrow, out_ncol = mod_selected_grid(image_grid, args['auto_square'])

    images_array = []
    for image in tqdm.tqdm(images, desc='Read Image'):
        images_array.append(np.array(Image.open(image)))
    images_array = np.array(images_array)
    images_array = np.transpose(images_array, (1, 2, 3, 0))

    grids = []
    for r in range(nrow):
        for c in range(ncol):
            r_start = r * grid_width
            r_end = r_start + grid_width
            c_start = c * grid_width
            c_end = c_start + grid_height
            grids.append(images_array[r_start:r_end, c_start:c_end])
    grids = np.array(grids)
    grid_index = np.where(image_grid.reshape(-1) == 1)[0]
    selected = np.array(grids[grid_index])

    results = []
    for i in tqdm.trange(out_nrow, desc='Make Output'):
        result = np.concatenate([*selected[out_ncol*i:out_ncol*(i+1)]], axis=1)
        results.append(result)
    results = np.transpose(np.concatenate(results, axis=0), (3, 0, 1, 2))

    basename = os.path.basename(os.path.dirname(images[0]))
    out_dir = os.path.join(args['output'], basename)
    os.makedirs(out_dir, exist_ok=True)
    for file_path, image in zip(images, tqdm.tqdm(results, desc='Write Image')):
        file_name = os.path.basename(file_path)
        out_path = os.path.join(out_dir, file_name)
        Image.fromarray(image).save(out_path)


if __name__ == '__main__':
    main()
