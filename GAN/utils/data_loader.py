import os
import tensorflow as tf


class ImageLoader:
    def __init__(self, data_txt_file, use_label=False):
        self.dataset = self._read_txt(data_txt_file, use_label)

    def _read_txt(self, txt, use_label):
        data_dir = os.path.dirname(txt)
        with open(txt, 'r', encoding='utf-8') as txt_file:
            if not use_label:
                dataset = tf.data.Dataset.from_tensor_slices([
                    os.path.join(data_dir, line.strip().split(',')[0])
                    for line in txt_file.readlines()])
            else:
                data = []
                label = []
                for line in txt_file.readlines():
                    line = line.strip().split(',')
                    file_path = os.path.join(data_dir, line[0])
                    data.append(file_path)
                    label.append(line[1])
                dataset = tf.data.Dataset.from_tensor_slices(
                    (data, label))
        return dataset

    def _read_file(self, data, label=None):
        data = tf.io.decode_png(tf.io.read_file(data))
        data = tf.cast(data, tf.float32)
        if label is None:
            return data
        return data, label

    def get_dataset(self,
                    batch_size,
                    map_func=None,
                    scailing=True,
                    new_size=None,
                    flatten=False,
                    shuffle=True,
                    drop_remainder=True):
        """
        new_size = (height, width)
        """
        dataset = self.dataset
        if shuffle:
            dataset = dataset.shuffle(len(self.dataset))

        dataset = dataset.map(
            map_func=self._read_file,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder
        )

        if (not scailing
            and map_func is None
            and new_size is None
                and not flatten):
            return dataset.cache(
            ).prefetch(tf.data.experimental.AUTOTUNE)

        def _total_map_func(data, label=None):
            if scailing:
                data = data / 127.5 - 1
            if new_size is not None:
                data = tf.image.resize(data, new_size)
            if flatten:
                data = tf.reshape(data, (batch_size, -1))
            if map_func is not None:
                data = map_func(data)
            if label is None:
                return data
            return data, label

        return dataset.map(
            map_func=_total_map_func,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).cache(
        ).prefetch(tf.data.experimental.AUTOTUNE)
