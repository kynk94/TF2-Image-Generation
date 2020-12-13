"""
Copyright (C) https://github.com/kynk94. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/).
"""
import os
import glob
import argparse
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from tensorflow.core.util import event_pb2
from util import DigitFirstSort, str_to_bool


def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('-l', '--log_dir', type=str,
                           default='./**/checkpoints',
                           help='Event log files directory, ' +
                           'Select exact log in runtime ' +
                           '(default=./**/checkpoints)')
    arg_parse.add_argument('-o', '--output', type=str,
                           default='./output/logs',
                           help='Output directory (default=./output/logs)')
    arg_parse.add_argument('-ei', '--extract_image', type=str_to_bool,
                           default=True,
                           help='Extract Image Flag (default=True)')
    args = vars(arg_parse.parse_args())

    logs = []
    for log_dir in glob.glob(args['log_dir']):
        logs.extend(glob.glob(os.path.join(log_dir, '**/events*'),
                              recursive=True))
    assert logs, 'Event log file not found'

    log_dirs = sorted({os.path.dirname(log)
                       for log in logs}, key=DigitFirstSort)
    print(f'Found {len(log_dirs)} train logs')

    if len(log_dirs) > 1:
        log_string = '\n'.join(f'\t{i}:\t{os.path.basename(log)}'
                               for i, log in enumerate(log_dirs))
        log_index = int(input(f'Found logs:\n{log_string}\n' +
                              'Select log (default: -1): ') or -1)
        logs = glob.glob(os.path.join(log_dirs[log_index], 'events*'))
    logs.sort()

    extract_image = args['extract_image']
    output_path = os.path.join(args['output'],
                               os.path.basename(os.path.dirname(logs[0])))

    scalar_output_path = os.path.join(output_path, 'scalar')
    os.makedirs(scalar_output_path, exist_ok=True)
    if extract_image:
        image_output_path = os.path.join(output_path, 'image')
        os.makedirs(image_output_path, exist_ok=True)
    else:
        image_output_path = None

    step = 0
    log_dict = defaultdict(dict)
    pbar = tqdm(logs, position=0, leave=True)
    for log in pbar:
        for event in tf.data.TFRecordDataset(log):
            event = event_pb2.Event.FromString(event.numpy())
            for value in event.summary.value:
                step = event.step
                plugin = value.metadata.plugin_data.plugin_name
                if plugin == 'scalars':
                    tensor = tf.make_ndarray(value.tensor).tolist()
                    log_dict[step][value.tag] = tensor
                elif extract_image and plugin == 'images':
                    tensor = tf.make_ndarray(value.tensor)[2]
                    image_name = os.path.join(image_output_path,
                                              f'{step:06d}.png')
                    tf.io.write_file(filename=image_name, contents=tensor)
            pbar.set_postfix({'step': step})

    data_frame = pd.DataFrame(log_dict).T
    data_frame.to_csv(os.path.join(scalar_output_path, 'log.csv'))


if __name__ == '__main__':
    main()
