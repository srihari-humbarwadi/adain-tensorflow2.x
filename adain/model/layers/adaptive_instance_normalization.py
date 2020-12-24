import tensorflow as tf


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.epsilon = kwargs.pop('epsilon', tf.keras.backend.epsilon())
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        style_images, content_images = inputs

        style_mean, style_variance = \
            tf.nn.moments(style_images, axes=[1, 2], keepdims=True)
        content_mean, content_variance = \
            tf.nn.moments(content_images, axes=[1, 2], keepdims=True)
        style_std = tf.sqrt(style_variance + self.epsilon)
        content_std = tf.sqrt(content_variance + self.epsilon)

        normalized_content_images = tf.math.divide_no_nan(
            content_images - content_mean, content_std)
        return style_std * normalized_content_images + style_mean

    def get_config(self):
        config = {'epsilon': self.epsilon}
        base_config = super(AdaptiveInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
