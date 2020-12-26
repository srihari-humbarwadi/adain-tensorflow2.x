import tensorflow as tf
from absl import app, flags, logging

from adain.cfg import Config
from adain.dataloader import InputPipeline
from adain.distribute import get_strategy
from adain.model import model_builder
from adain.trainer import Trainer

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)

flags.DEFINE_string('config_path',
                    default=None,
                    help='Path to the config file')

flags.DEFINE_boolean('xla', default=False, help='Compile with XLA JIT')

flags.DEFINE_boolean('gpu_memory_allow_growth',
                     default=False,
                     help='If enabled, the runtime doesn\'t allocate all of the available memory')  # noqa: E501

flags.DEFINE_boolean('debug', default=False, help='Print debugging info')

FLAGS = flags.FLAGS


def set_precision(precision):
    policy = tf.keras.mixed_precision.Policy(precision)
    tf.keras.mixed_precision.set_global_policy(policy)

    logging.info('Compute dtype: {}'.format(policy.compute_dtype))
    logging.info('Variable dtype: {}'.format(policy.variable_dtype))


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    if FLAGS.xla:
        tf.config.optimizer.set_jit(True)

    if FLAGS.gpu_memory_allow_growth:
        physical_devices = tf.config.list_physical_devices('GPU')
        [tf.config.experimental.set_memory_growth(x, True)
         for x in physical_devices]

    params = Config(FLAGS.config_path).params

    set_precision(params.floatx.precision)
    strategy = get_strategy(params.training.strategy)

    train_dataset_fn = InputPipeline(params, is_validation_dataset=False)
    val_train_dataset_fn = InputPipeline(params, is_validation_dataset=True)

    model_fn = model_builder(params)

    trainer = Trainer(strategy=strategy,  # noqa: F841
                      model_fn=model_builder(params),
                      train_input_fn=train_dataset_fn,
                      val_input_fn=val_train_dataset_fn,
                      train_steps=params.training.train_steps,
                      val_steps=params.training.validation_steps,
                      val_freq=params.training.validation_freq,
                      steps_per_execution=params.training.steps_per_execution,
                      batch_size=params.training.batch_size,
                      model_dir=params.experiment.model_dir,
                      save_every=params.training.save_every,
                      restore_checkpoint=params.training.restore_checkpoint,
                      summary_dir=params.experiment.tensorboard_dir,
                      name=params.experiment.name)

if __name__ == '__main__':
    app.run(main)
