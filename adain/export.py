import tensorflow as tf
from absl import app, flags, logging

from adain.cfg import Config
from adain.model import model_builder
from adain.trainer import Trainer

tf.get_logger().propagate = False
tf.config.set_soft_device_placement(True)

flags.DEFINE_string('config_path',
                    default=None,
                    help='Path to the config file')

flags.DEFINE_string('export_dir',
                    default='export',
                    help='Path to store the `saved_model`')


flags.DEFINE_boolean('debug', default=False, help='Print debugging info')

FLAGS = flags.FLAGS


def main(_):
    logging.set_verbosity(logging.DEBUG if FLAGS.debug else logging.INFO)

    params = Config(FLAGS.config_path).params

    trainer = Trainer(run_mode='export',  # noqa: F841
                      strategy=tf.distribute.OneDeviceStrategy(device='/cpu:0'),  # noqa: E501
                      model_fn=model_builder(params),
                      train_input_fn=None,
                      val_input_fn=None,
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

    inference_model = trainer.model
    inference_model.build([(None, None, None, 3), (None, None, None, 3), ()])
    inference_model.optimizer = None
    inference_model.compiled_loss = None
    inference_model.compiled_metrics = None
    inference_model._metrics = []

    input_signature = [(
        tf.TensorSpec(shape=[None, None, None, 3],
                      name='style_images',
                      dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, 3],
                      name='content_images',
                      dtype=tf.float32),
        tf.TensorSpec(shape=[], name='alpha', dtype=tf.float32),
    )]
    
    logging.debug('Signature for serving_default :\n{}'.format(
        input_signature))
    
    inference_model._saved_model_inputs_spec = input_signature

    @tf.function(input_signature=input_signature)
    def serving_fn(input_data):
        style_images, content_images, alpha = input_data

        generated_images = inference_model.call(
            (style_images, content_images, alpha),
            training=False)['synthesized_images']

        return {'generated_images': generated_images}

    logging.info('Exporting `saved_model` to {}'.format(FLAGS.export_dir))

    tf.saved_model.save(
        inference_model,
        FLAGS.export_dir,
        signatures={'serving_default': serving_fn.get_concrete_function()})  # noqa: E501

if __name__ == '__main__':
    app.run(main)
