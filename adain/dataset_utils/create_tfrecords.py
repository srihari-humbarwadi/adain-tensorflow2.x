import os
from glob import glob
from random import shuffle

import tensorflow as tf
from absl import app, flags, logging
from adain.dataset_utils.tfrecord_writer import TFrecordWriter

flags.DEFINE_string('image_paths_pattern',
                    default=None,
                    help='File pattern matching all image names')

flags.DEFINE_string('prefix',
                    default='images',
                    help='Prefix for generated tfrecord files')

flags.DEFINE_integer('num_shards',
                     default=256,
                     help='Number of tfrecord files required.')

flags.DEFINE_integer('num_images',
                     default=-1,
                     help='Number of images to use')

flags.DEFINE_string('output_dir',
                    default='./tfrecords',
                    help='Path to store the generated tfrecords in.')

FLAGS = flags.FLAGS


def write_tfrecords(image_paths, num_shards, output_dir, prefix):
    tfrecord_writer = TFrecordWriter(n_samples=len(image_paths),
                                     n_shards=num_shards,
                                     output_dir=output_dir,
                                     prefix=prefix)
    bad_samples = 0
    for image_path in image_paths:
        try:
            with tf.io.gfile.GFile(image_path, 'rb') as fp:
                image = fp.read()
                h, w, _ = tf.image.decode_image(image).shape.as_list()
        except Exception:
            bad_samples += 1
            continue

        tfrecord_writer.push(image)
    tfrecord_writer.flush_last()
    logging.warning('Skipped {} corrupted samples from {} data'.format(
        bad_samples, prefix))


def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    image_paths = sorted(glob(FLAGS.image_paths_pattern))

    logging.info('Found {} matching images with the pattern: {}'.format(
        len(image_paths), FLAGS.image_paths_pattern))

    shuffle(image_paths)

    if FLAGS.num_images != -1:
        image_paths = image_paths[:FLAGS.num_images]
        logging.info('Using {} images from {} total images'.format(
            FLAGS.num_images, len(image_paths)))

    write_tfrecords(image_paths, FLAGS.num_shards,
                    FLAGS.output_dir, FLAGS.prefix)


if __name__ == '__main__':
    app.run(main)
