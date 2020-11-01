import os
import shutil
from abc import ABC
from datetime import datetime

import tensorflow as tf


class BaseModel(ABC):
    def __init__(self, conf, ckpt=None):
        self.conf = conf
        self.ckpt = None
        self.ckpt_file = None
        self.ckpt_manager = None
        self._conf_path = None
        self._checkpoint_dir = None
        self._output_dir = None

        self._set_dirs(self.load(ckpt))
        self._logger = self._create_logger()

    def _set_dirs(self, time_stamp=None):
        if time_stamp is None:
            now = datetime.now().strftime('%y-%m-%d_%H_%M_%S')
            time_stamp = f'{self.__class__.__name__}_{now}'
        self._checkpoint_dir = os.path.join(self.conf['checkpoint_dir'],
                                            time_stamp)
        self._output_dir = os.path.join(self._checkpoint_dir, 'output')

    def set_checkpoint(self, max_to_keep=5, **kwargs):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                        **kwargs)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       self._checkpoint_dir,
                                                       max_to_keep=max_to_keep)
        if self.ckpt_file is not None:
            self.ckpt.restore(self.ckpt_file)

    def image_write(self, filename, data, denorm=True):
        if denorm:
            data = data * 127.5 + 127.5
        os.makedirs(self._output_dir, exist_ok=True)
        tf.io.write_file(filename=os.path.join(self._output_dir, filename),
                         contents=tf.io.encode_png(tf.cast(data, tf.uint8)))

    def copy_conf(self, conf_path):
        shutil.copy(conf_path, self._checkpoint_dir)

    def save(self):
        self.ckpt_manager.save(checkpoint_number=self.ckpt.step)

    def load(self, checkpoint_path):
        if checkpoint_path is None:
            return

        ckpt = None
        if os.path.isdir(checkpoint_path):
            ckpt = tf.train.latest_checkpoint(checkpoint_path)
        elif os.path.exists(checkpoint_path + '.index'):
            ckpt = checkpoint_path

        if ckpt is None:
            raise FileNotFoundError('checkpoint_path not found.')

        self.ckpt_file = ckpt
        return os.path.basename(os.path.dirname(ckpt))

    def _create_logger(self):
        return tf.summary.create_file_writer(self._checkpoint_dir)

    def write_scalar_log(self, **kwargs):
        step = self.ckpt.step
        with self._logger.as_default():
            for name, data in kwargs.items():
                tf.summary.scalar(name=name, data=data, step=step)

    def write_image_log(self, step, data, name='output', denorm=True):
        if len(data.shape) == 3:
            data = tf.expand_dims(data, axis=0)
        if denorm:
            data = data / 2 + 0.5
        with self._logger.as_default():
            tf.summary.image(name=name, data=data, step=step)
