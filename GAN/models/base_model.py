import os
import shutil
from abc import ABC
from datetime import datetime

import tensorflow as tf


class BaseModel(ABC):
    def __init__(self, conf):
        self.conf = conf
        self.ckpt = None
        self.ckpt_manager = None
        self._conf_path = None
        self._checkpoint_dir = None
        self._output_dir = None
        self._set_dirs()
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

    def image_write(self, filename, data, denorm=True):
        if denorm:
            data = data * 127.5 + 127.5
        tf.io.write_file(filename=os.path.join(self._output_dir, filename),
                         contents=tf.io.encode_png(tf.cast(data, tf.uint8)))

    def copy_conf(self, conf_path):
        self._conf_path = conf_path
        shutil.copy(conf_path, self._checkpoint_dir)

    def save(self):
        self.ckpt_manager.save(checkpoint_number=self.ckpt.step)

    def load(self, checkpoint_path, new_log=False):
        ckpt = None
        if os.path.isdir(checkpoint_path):
            ckpt = tf.train.latest_checkpoint(checkpoint_path)
        elif os.path.exists(checkpoint_path + '.index'):
            ckpt = checkpoint_path

        if ckpt is None:
            raise FileNotFoundError('checkpoint_path not found.')

        if not new_log:
            self._set_dirs(os.path.basename(os.path.dirname(ckpt)))
            if self._conf_path is not None:
                current_conf = os.path.join(self._checkpoint_dir,
                                            self._conf_path)
                if os.path.exists(current_conf):
                    os.remove(current_conf)
                self.copy_conf(self._conf_path)
            self._logger = self._create_logger()

        self.ckpt.restore(ckpt)
        return self.ckpt.step

    def _create_logger(self):
        return tf.summary.create_file_writer(self._checkpoint_dir)
