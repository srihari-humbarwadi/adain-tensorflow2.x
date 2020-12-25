import tensorflow as tf


class AdaptiveInstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, alpha=1.0, **kwargs):
        self.epsilon = kwargs.pop('epsilon', tf.keras.backend.epsilon())
        super(AdaptiveInstanceNormalization, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        style_features, content_features = inputs

        style_mean, style_variance = \
            tf.nn.moments(style_features, axes=[1, 2], keepdims=True)
        content_mean, content_variance = \
            tf.nn.moments(content_features, axes=[1, 2], keepdims=True)
        style_std = tf.sqrt(style_variance + self.epsilon)
        content_std = tf.sqrt(content_variance + self.epsilon)

        normalized_content_features = tf.math.divide_no_nan(
            content_features - content_mean, content_std)
        output = style_std * normalized_content_features + style_mean
        return self.alpha * output + (1 - self.alpha) * content_features

    def get_config(self):
        config = {'alpha': self.alpha, 'epsilon': self.epsilon}
        base_config = super(AdaptiveInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
