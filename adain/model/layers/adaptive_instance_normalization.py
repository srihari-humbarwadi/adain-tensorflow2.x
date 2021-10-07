import tensorflow as tf


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        self.epsilon = kwargs.pop('epsilon', tf.keras.backend.epsilon())
        super(AdaptiveInstanceNormalization, self).__init__(
            dtype=tf.float32, **kwargs)

    def call(self, inputs):
        style_features, content_features = inputs

        style_mean, style_variance = \
            tf.nn.moments(style_features, axes=[1, 2], keepdims=True)
        content_mean, content_variance = \
            tf.nn.moments(content_features, axes=[1, 2], keepdims=True)

        output = tf.nn.batch_normalization(
            x=content_features,
            mean=content_mean,
            variance=content_variance,
            offset=style_mean,
            scale=tf.sqrt(style_variance),
            variance_epsilon=self.epsilon)
        return output

    def get_config(self):
        config = {'epsilon': self.epsilon}
        base_config = super(AdaptiveInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
