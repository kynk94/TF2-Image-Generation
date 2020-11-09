import re
import argparse
import tensorflow as tf


def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def digit_first_sort(string):
    digits = []
    strings = []
    for s in re.split(r'(\d+)', string):
        try:
            digits.append(int(s))
        except:
            strings.append(s)
    digits.extend(strings)
    return digits


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    if value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def float_0_to_1(value):
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a floating point number')
    if 0.0 < value < 1.0:
        return value
    raise argparse.ArgumentTypeError('Must be in range(0.0, 1.0)')
