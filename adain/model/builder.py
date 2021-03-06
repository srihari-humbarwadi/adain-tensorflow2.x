import tensorflow as tf
from absl import logging

from adain.learning_rate_schedule import InverseDecay
from adain.model import StyleTransferNetwork


def get_optimizer(name):
    if name == 'sgd':
        return tf.optimizers.SGD
    if name == 'adam':
        return tf.optimizers.Adam
    raise ValueError('Unsupported optimizer requested')


def model_builder(params):
    def _model_fn():
        model = StyleTransferNetwork()

        logging.info('Trainable weights: {}'.format(
            len(model.trainable_weights)))

        learning_rate_params = params.training.optimizer.learning_rate
        learning_rate_fn = InverseDecay(
            learning_rate_params.initial_learning_rate,
            learning_rate_params.decay_rate)

        optimizer = get_optimizer(params.training.optimizer.name)(
            learning_rate=learning_rate_fn)

        if params.floatx.precision == 'mixed_float16':
            logging.info(
                'Wrapping optimizer with `LossScaleOptimizer` for AMP training'
            )
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer, dynamic=True)

        model.compile(optimizer=optimizer)

        model.encoder.summary(print_fn=logging.debug, line_length=80)
        logging.debug('\n')
        model.decoder.summary(print_fn=logging.debug, line_length=80)

        logging.info('Total trainable parameters: {:,}'.format(
            sum([
                tf.keras.backend.count_params(x)
                for x in model.trainable_variables
            ])))
        return model

    return _model_fn
