import tensorflow as tf


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
