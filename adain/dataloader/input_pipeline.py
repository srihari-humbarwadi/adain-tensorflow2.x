import tensorflow as tf
from absl import logging

from adain.dataloader.preprocessing_pipeline import PreprocessingPipeline
from adain.dataloader.tfrecord_parser import parse_example


class InputPipeline:

    def __init__(self, params, is_validation_dataset):
        self.is_validation_dataset = is_validation_dataset
        self.batch_size = params.training.batch_size
        self.tfrecord_files = params.dataloader_params.tfrecords[
            'train' if not is_validation_dataset else 'val']
        self.preprocessing_pipeline = PreprocessingPipeline(params, is_validation_dataset)

    def __call__(self, input_context=None):
        options = tf.data.Options()
        options.experimental_deterministic = False
        autotune = tf.data.experimental.AUTOTUNE

        style_dataset = tf.data.Dataset.list_files(self.tfrecord_files['style'])
        content_dataset = tf.data.Dataset.list_files(
            self.tfrecord_files['content'])

        logging.info('Found {} style tfrecords matching {}'.format(
            len(style_dataset), self.tfrecord_files['style']))

        logging.info('Found {} content tfrecords matching {}'.format(
            len(content_dataset), self.tfrecord_files['content']))

        style_dataset = style_dataset.cache()
        content_dataset = content_dataset.cache()

        if not self.is_validation_dataset:
            style_dataset = style_dataset.repeat()
            content_dataset = content_dataset.repeat()

        style_dataset = style_dataset.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=32,
            num_parallel_calls=autotune)

        content_dataset = content_dataset.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=32,
            num_parallel_calls=autotune)

        style_dataset = style_dataset.with_options(options)
        content_dataset = content_dataset.with_options(options)
        style_dataset = style_dataset.shuffle(1024)
        content_dataset = content_dataset.shuffle(1024)

        style_dataset = style_dataset.map(
            map_func=lambda x: self.preprocessing_pipeline(parse_example(x)),
            num_parallel_calls=autotune)

        content_dataset = content_dataset.map(
            map_func=lambda x: self.preprocessing_pipeline(parse_example(x)),
            num_parallel_calls=autotune)

        dataset = tf.data.Dataset.zip((style_dataset, content_dataset))
        
        if not self.is_validation_dataset:
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)  # noqa: E501
            
        dataset = dataset.prefetch(autotune)
        return dataset
