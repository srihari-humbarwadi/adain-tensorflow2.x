import tensorflow as tf


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.epsilon = 1e-5
        super(AdaptiveInstanceNormalization, self).__init__(
            dtype=tf.float32, **kwargs)

    def call(self, inputs):
        style_features, content_features = inputs
        style_mean, style_variance = \
            tf.nn.moments(style_features, axes=[1, 2], keepdims=True)
        content_mean, content_variance = \
            tf.nn.moments(content_features, axes=[1, 2], keepdims=True)
        style_std = tf.sqrt(style_variance + self.epsilon)
        content_std = tf.sqrt(content_variance + self.epsilon)

        normalized_content_features = (
            content_features - content_mean) / content_std
        output = style_std * normalized_content_features + style_mean
        return output

    def get_config(self):
        config = {'epsilon': self.epsilon}
        base_config = super(AdaptiveInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
