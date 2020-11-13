import os
import tensorflow as tf
from collections import defaultdict
from collections.abc import Iterable


class ImageLoader:
    def __init__(self, data_txt_file, use_label=False):
        self.n_data = None
        self.n_class = None
        self.class_dict = defaultdict(ClassCounter())
        self.class_dict_pair = None
        self.dataset = self._read_txt(data_txt_file, use_label)

    def _read_txt(self, txt, use_label):
        data_dir = os.path.dirname(txt)
        with open(txt, 'r', encoding='utf-8') as txt_file:
            if not use_label:
                data = [os.path.join(data_dir, line.strip().split(',')[0])
                        for line in txt_file.readlines()]
                dataset = tf.data.Dataset.from_tensor_slices(data)
            else:
                data = []
                labels = []
                for line in txt_file.readlines():
                    path, label = line.strip().split(',')
                    file_path = os.path.join(data_dir, path)
                    data.append(file_path)
                    labels.append(self.class_dict[label])
                dataset = tf.data.Dataset.from_tensor_slices(
                    (data, labels))
        self.n_data = len(data)
        self.n_class = len(self.class_dict)
        self.class_dict_pair = dict(zip(self.class_dict.values(),
                                        self.class_dict.keys()))
        return dataset

    def _read_file(self, data, label=None, new_size=None):
        data = tf.io.decode_png(tf.io.read_file(data), channels=3)
        data = tf.cast(data, tf.float32)
        if new_size is not None:
            data = tf.image.resize(data, new_size)
        data = tf.transpose(data, perm=(2, 0, 1))
        if label is None:
            return data
        label = tf.cast(label, tf.float32)
        return data, label

    def get_dataset(self,
                    batch_size,
                    map_func=None,
                    scailing=True,
                    new_size=None,
                    flatten=False,
                    shuffle=True,
                    drop_remainder=True,
                    cache=True):
        """
        new_size = (height, width)
        """
        dataset = self.dataset
        if shuffle:
            dataset = dataset.shuffle(self.n_data)

        dataset = dataset.map(
            map_func=lambda x, y=None: self._read_file(
                x, label=y, new_size=new_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder
        )

        if (not scailing
            and map_func is None
            and new_size is None
                and not flatten):
            if cache:
                dataset = dataset.cache()
            return dataset.prefetch(tf.data.experimental.AUTOTUNE)

        def _total_map_func(data, label=None):
            if scailing:
                data = data / 127.5 - 1
            if flatten:
                data = tf.reshape(data, (batch_size, -1))
            if map_func is not None:
                data = map_func(data)
            if label is None:
                return data
            return data, label

        dataset = dataset.map(
                map_func=_total_map_func,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if cache:
            dataset = dataset.cache()
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def get_label(self, str_label=None, num_label=None):
        if str_label is not None:
            if isinstance(str_label, Iterable):
                return [self.class_dict[s] for s in str_label]
            return self.class_dict[str_label]

        if num_label is not None:
            if isinstance(num_label, Iterable):
                return [self.class_dict_pair[n] for n in num_label]
            return self.class_dict_pair[num_label]

        raise ValueError('Should input either str_label or num_label.')

    def __len__(self):
        return self.n_data


class ClassCounter:
    count = -1

    def __call__(self):
        self.count += 1
        return self.count
