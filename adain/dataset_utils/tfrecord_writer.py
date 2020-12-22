import os

import tensorflow as tf
from absl import logging


class TFrecordWriter:
    def __init__(self, n_samples, n_shards, output_dir='', prefix=''):
        self.n_samples = n_samples
        self.n_shards = n_shards
        self._step_size = self.n_samples // self.n_shards + 1
        self.prefix = prefix
        self.output_dir = output_dir
        self._buffer = []
        self._file_count = 1

    def _make_example(self, image):
        feature = {
            'image':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def _write_tfrecord(self, tfrecord_path):
        logging.info('writing {} samples in {}'.format(len(self._buffer),
                                                       tfrecord_path))
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for (image,) in self._buffer:
                example = self._make_example(image)
                writer.write(example.SerializeToString())

    def push(self, image):
        self._buffer.append((image, ))
        if len(self._buffer) == self._step_size:
            fname = self.prefix + '-{:04.0f}'.format(
                self._file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self._write_tfrecord(tfrecord_path)
            self._clear_buffer()
            self._file_count += 1

    def flush_last(self):
        if self._buffer:
            fname = self.prefix + '-{:04.0f}'.format(
                self._file_count) + '.tfrecord'
            tfrecord_path = os.path.join(self.output_dir, fname)
            self._write_tfrecord(tfrecord_path)

    def _clear_buffer(self):
        self._buffer = []
