import os
from time import time

import numpy as np
import tensorflow as tf
from absl import logging

from adain.viz_utils import prepare_visualization_image


class Trainer:

    _RUN_MODES = [
        'train',
        'train_val',
        'export'
    ]

    def __init__(self,
                 run_mode,
                 strategy,
                 model_fn,
                 style_loss_weight,
                 content_loss_weight,
                 train_input_fn=None,
                 val_input_fn=None,
                 train_steps=None,
                 val_steps=None,
                 val_freq=None,
                 steps_per_execution=250,
                 batch_size=None,
                 model_dir='./model_files',
                 save_every=1000,
                 restore_checkpoint=True,
                 summary_dir='tensorboard',
                 name=None):
        self.run_mode = run_mode
        self.distribute_strategy = strategy
        self.model_fn = model_fn
        self.train_input_fn = train_input_fn
        self.val_input_fn = val_input_fn
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.val_freq = val_freq
        self.steps_per_execution = steps_per_execution
        self.batch_size = batch_size
        self.model_dir = os.path.join(model_dir, name)
        self.save_every = save_every
        self.restore_checkpoint = restore_checkpoint
        self.summary_dir = summary_dir
        self.name = name
        self.restore_status = None
        self.use_float16 = False
        self.training_alpha = tf.constant(1.0, dtype=tf.float32)
        self.style_loss_weight = tf.constant(style_loss_weight,
                                             dtype=tf.float32)
        self.content_loss_weight = tf.constant(content_loss_weight,
                                               dtype=tf.float32)

        assert self.run_mode in Trainer._RUN_MODES, \
            'Invalid run mode, aborting!\n Supported run modes {}' \
            .format(Trainer._RUN_MODES)

        self._setup()

        if 'train' in self.run_mode:
            self.train()

    def _setup_model(self):
        logging.info('Setting up model')
        self._model = self.model_fn()
        self.optimizer = self._model.optimizer
        if isinstance(self.optimizer,
                      tf.keras.mixed_precision.LossScaleOptimizer):
            self.use_float16 = True

    def _setup_dataset(self):
        if 'train' in self.run_mode:
            logging.info('Setting up train dataset')
            self._train_dataset = \
                self.distribute_strategy.experimental_distribute_dataset(
                    self.train_input_fn())

        if 'val' in self.run_mode:
            logging.info('Setting up val dataset')
            self._val_dataset = self.val_input_fn()

    def _setup_summary_writer(self):
        logging.info('Setting up summary writer')
        self._summary_writer = tf.summary.create_file_writer(
            os.path.join(self.summary_dir, self.name))
        self._summary_writer.set_as_default()
        logging.info('Writing summaries to {}'.format(
            os.path.join(self.summary_dir, self.name)))

    def _restore_checkpoint(self):
        logging.info('Looking for existing checkpoints in {}'.format(
            self.model_dir))
        latest_checkpoint = tf.train.latest_checkpoint(self.model_dir)
        if latest_checkpoint is not None:
            logging.info(
                'Found for existing checkpoint {}, restoring model and optimizer state from checkpoint'  # noqa: E501
                .format(latest_checkpoint))
            self.optimizer._create_all_weights(self._model.trainable_variables)
            self.restore_status = self._model.load_weights(latest_checkpoint)
            return

        logging.info(
            'No existing checkpoints found in {}, training from scratch!'.
            format(self.model_dir))

    def _setup(self):
        with self.distribute_strategy.scope():
            self._setup_model()
            self._setup_dataset()

            if self.restore_checkpoint:
                self._restore_checkpoint()

    @tf.function
    def _write_summaries(self, loss_dict, step):
        with self._summary_writer.as_default():

            with tf.name_scope('losses'):
                for k in ['content-loss', 'style-loss', 'total-loss']:
                    v = loss_dict[k]
                    tf.summary.scalar(k, data=v, step=step)

            with tf.name_scope('metrics'):
                for k in ['learning-rate', 'execution-time']:
                    v = loss_dict[k]
                    tf.summary.scalar(k, data=v, step=step)

    def _train_step(self, data):
        style_images, content_images = data
        training_input = (style_images['image'], content_images['image'],
                          self.training_alpha)
        with tf.GradientTape() as tape:
            outputs = self._model(training_input, training=True)
            loss = outputs['loss']
            style_loss_weight = tf.cast(self.style_loss_weight,
                                        dtype=loss['content-loss'].dtype)
            content_loss_weight = tf.cast(self.content_loss_weight,
                                          dtype=loss['content-loss'].dtype)
            loss['total-loss'] = (
                content_loss_weight * loss['content-loss'] +
                style_loss_weight * loss['style-loss'])

            per_replica_loss = loss[
                'total-loss'] / self.distribute_strategy.num_replicas_in_sync
            if self.use_float16:
                per_replica_loss = self.optimizer.get_scaled_loss(
                    per_replica_loss)

        gradients = tape.gradient(per_replica_loss,
                                  self._model.trainable_variables)
        if self.use_float16:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(
            zip(gradients, self._model.trainable_variables))
        return loss

    @tf.function
    def distributed_train_step(self, iterator, num_steps):
        loss_dict = self.distribute_strategy.run(self._train_step,
                                                 args=(next(iterator),))
        for _ in tf.range(num_steps - 1):
            loss_dict = self.distribute_strategy.run(self._train_step,
                                                     args=(next(iterator),))
        loss_dict = tf.nest.map_structure(
            lambda x: self.distribute_strategy.reduce(
                tf.distribute.ReduceOp.MEAN, x, axis=None), loss_dict)
        return loss_dict

    def train(self):
        if self.restore_checkpoint and self.restore_status is not None:
            self.restore_status.assert_consumed()

        start_step = int(self.optimizer.iterations.numpy())
        current_step = start_step

        if current_step == self.train_steps:
            logging.info('Training completed at step {}'.format(current_step))
            return

        logging.info(
            'Starting training from step {} for {} steps with {} steps per execution'  # noqa: E501
            .format(start_step, self.train_steps, self.steps_per_execution))
        logging.info('Saving checkpoints every {} steps in {}'.format(
            self.save_every, self.model_dir))

        self._setup_summary_writer()

        if self.use_float16:
            logging.info('Training with AMP turned on!')

        dataset_iterator = iter(self._train_dataset)
        for _ in range(start_step, self.train_steps, self.steps_per_execution):
            start = time()
            loss_dict = self.distributed_train_step(
                dataset_iterator,
                tf.convert_to_tensor(self.steps_per_execution))
            current_step = int(self.optimizer.iterations.numpy())
            end = time()

            if self.use_float16:
                learning_rate = self.optimizer._optimizer._decayed_lr('float')
            else:
                learning_rate = self.optimizer._decayed_lr('float')

            loss_dict['execution-time'] = np.round(end - start, 2)
            loss_dict['learning-rate'] = learning_rate

            per_step_execution_time = \
                self.steps_per_execution / loss_dict['execution-time']
            images_per_second = per_step_execution_time * self.batch_size

            secs = (self.train_steps - current_step) / per_step_execution_time
            eta = []
            for interval in [3600, 60, 1]:
                eta += ['{:02}'.format(int(secs // interval))]
                secs %= interval
            eta = ':'.join(eta)

            if current_step % self.save_every == 0:
                logging.info(
                    'Saving checkpoint at step {}'.format(current_step))
                self._model.save_weights(
                    os.path.join(self.model_dir,
                                 'weights_step_{}'.format(current_step)))

            self._write_summaries(
                loss_dict, tf.convert_to_tensor(current_step, dtype=tf.int64))

            logging.info(
                '[global_step {}/{}] [ETA: {}] [{:.2f} imgs/s] {}'.format(
                    current_step, self.train_steps, eta, images_per_second,
                    {k: np.round(v, 3) for k, v in loss_dict.items()}))

            if (current_step % self.val_freq == 0) and ('val' in self.run_mode):
                logging.info('Generating outputs with validation images')

                visualization_image = []

                for idx, (style_image, content_image) in enumerate(
                        self._val_dataset.take(self.val_steps)):

                    generated_image = self._model(
                        (style_image['image'][None, ...],
                         content_image['image'][None, ...],
                         self.training_alpha),
                        training=False)['synthesized_images'][0]

                    visualization_image += [prepare_visualization_image(
                        style_image['image'], content_image['image'],
                        generated_image)]

                visualization_image = tf.concat(visualization_image, axis=1)
                with self._summary_writer.as_default():
                    tf.summary.image('Generated Images',
                                     visualization_image,
                                     step=tf.convert_to_tensor(
                                         current_step, dtype=tf.int64))

        logging.info('Saving final checkpoint at step {}'.format(current_step))
        self._model.save_weights(
            os.path.join(self.model_dir,
                         'final_weights_step_{}'.format(current_step)))

    @property
    def model(self):
        return self._model
