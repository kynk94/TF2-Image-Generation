import os
from abc import ABC, abstractmethod
from datetime import datetime

import tensorflow as tf


class BaseModel(ABC):
    def __init__(self, conf):
        self.conf = conf
        self.checkpoint = None
        self._output_dir = None
        self._checkpoint_dir = None
        self._log_dir = None
        self._set_dirs()

    def _set_dirs(self, time_stamp=None):
        if time_stamp is None:
            now = datetime.now().strftime('%y-%m-%d_%H_%M_%S')
            time_stamp = f'{self.__class__.__name__}_{now}'
        self._output_dir = os.path.join(self.conf['output_dir'], time_stamp)
        self._checkpoint_dir = os.path.join(self.conf['checkpoint_dir'],
                                            time_stamp)
        self._log_dir = os.path.join(self.conf['log_dir'], time_stamp)

    @abstractmethod
    def _set_checkpoint(self):
        pass

    def save(self):
        self.checkpoint.save(file_prefix=os.path.join(self._checkpoint_dir,
                                                      'ckpt'))

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
        self.checkpoint.restore(ckpt)
        return getattr(self.checkpoint, 'step', None)

    def _create_logger(self):
        return tf.summary.create_file_writer(self._log_dir)
