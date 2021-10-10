import tensorflow as tf
from absl import app, flags, logging
from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2_as_graph

from adain.cfg import Config
from adain.dataloader.preprocessing_pipeline import PreprocessingPipeline
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

    trainer = Trainer(
        run_mode='export',  # noqa: F841
        strategy=tf.distribute.OneDeviceStrategy(device='/cpu:0'),  # noqa: E501
        model_fn=model_builder(params),
        train_input_fn=None,
        val_input_fn=None,
        style_loss_weight=params.training.style_loss_weight,
        content_loss_weight=params.training.content_loss_weight,
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
    inference_model.build([(1, None, None, 3), (1, None, None, 3), ()])
    inference_model.optimizer = None
    inference_model.compiled_loss = None
    inference_model.compiled_metrics = None
    inference_model._metrics = []

    input_signature = {
        'style_images': tf.TensorSpec(
            shape=[1, None, None, 3],
            name='style_images',
            dtype=tf.float32),
        'content_images': tf.TensorSpec(
            shape=[1, None, None, 3],
            name='content_images',
            dtype=tf.float32),
        'alpha': tf.TensorSpec(
            shape=[], name='alpha', dtype=tf.float32)
    }

    logging.debug('Signature for serving_default :\n{}'.format(
        input_signature))

    inference_model._saved_model_inputs_spec = input_signature

    preprocessing_pipeline = PreprocessingPipeline(params=params,
                                                   is_validation_dataset=True)

    @tf.function
    def serving_fn(sample):
        style_images = preprocessing_pipeline.inference_pipeline(
            sample['style_images'])
        content_images = preprocessing_pipeline.inference_pipeline(
            sample['content_images'])

        stylized_images = inference_model.call((
            style_images,
            content_images,
            sample['alpha']),
            training=False)['synthesized_images']

        stylized_images = preprocessing_pipeline.denormalize(stylized_images)
        return {'stylized_images': stylized_images}

    frozen_serving_fn, _ = convert_variables_to_constants_v2_as_graph(
        serving_fn.get_concrete_function(input_signature),
        aggressive_inlining=True)

    class InferenceModule(tf.Module):
        def __init__(self, inference_function):
            super(InferenceModule, self).__init__(name='inference_module')
            self.inference_function = inference_function

        @tf.function
        def run_inference(self, sample):
            outputs = self.inference_function(**sample)
            return {'stylized_images': outputs[0]}

    inference_module = InferenceModule(inference_function=frozen_serving_fn)
    signatures = {
        'serving_default': inference_module.run_inference.get_concrete_function(
            input_signature)
    }
    print(signatures['serving_default'].output_shapes)
    for _signature_name, _concrete_fn in signatures.items():
        input_shapes = {x.name.split(':')[0]: x.shape.as_list()
                        for x in _concrete_fn.inputs}

        output_shapes = {k: v.as_list()
                         for k, v in _concrete_fn.output_shapes.items()}
        logging.info(
            '\nSignature: {}\n Input Shapes:\n {}\nOutput Shapes:\n{}'.format(
                _signature_name,
                input_shapes,
                output_shapes))

    logging.info('Exporting `saved_model` to {}'.format(FLAGS.export_dir))
    tf.saved_model.save(
        obj=inference_module,
        export_dir=FLAGS.export_dir,
        signatures=signatures,
        options=tf.saved_model.SaveOptions(
            experimental_custom_gradients=False))


if __name__ == '__main__':
    app.run(main)
