import tensorflow as tf


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        style_images, content_images = inputs

        style_mean, style_variance = \
            tf.nn.moments(style_images, axes=[1, 2], keepdims=True)
        content_mean, content_variance = \
            tf.nn.moments(content_images, axes=[1, 2], keepdims=True)
        style_std = tf.sqrt(style_variance + tf.keras.backend.epsilon())
        content_std = tf.sqrt(content_variance + tf.keras.backend.epsilon())

        normalized_content_images = tf.math.divide_no_nan(
            content_images - content_mean, content_std)
        return style_std * normalized_content_images + style_mean

    def get_config(self):
        return super(AdaptiveInstanceNormalization, self).get_config()


class ReflectionPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)

        self.paddings = ((0, 0), (padding, padding), (padding, padding), (0, 0))

    def call(self, x):
        return tf.pad(x, paddings=self.paddings, mode='REFLECT')

    def get_config(self):
        config = {'paddings': self.paddings}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
