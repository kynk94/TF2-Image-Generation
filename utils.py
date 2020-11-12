import re
import argparse
import operator
from itertools import zip_longest


class DigitFirstSort:
    def __init__(self, obj, *args):
        self.obj = self._split_digit_string(obj)

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def _split_digit_string(obj):
        digits = []
        strings = []
        for i in re.split(r'(\d+)', obj):
            if not i:
                continue
            if i.isdigit():
                digits.append(int(i))
            else:
                strings.append(i)
        digits.extend(strings)
        return digits

    def _compare(self, other, operation):
        for i, j in zip_longest(self.obj, other.obj):
            if type(i) == type(j):
                if i == j:
                    continue
                elif operation(i, j):
                    continue
            else:
                s_i, s_j = str(i), str(j)
                if s_i == s_j:
                    continue
                elif operation(s_i, s_j):
                    continue
            return False
        return True


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
