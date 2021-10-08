import functools

from absl import logging
import tensorflow as tf

from adain.model.layers import AdaptiveInstanceNormalization  # noqa E501
from adain.model.layers import ReflectionPadding2D


def get_vgg_model(preprocessing_params):
    vgg_old = tf.keras.applications.VGG19(
        input_shape=[None, None, 3],
        weights=None,
        include_top=False)

    images = tf.keras.Input(shape=[None, None, 3])
    offset = tf.reshape(tf.constant(preprocessing_params.offset),
                        shape=[1, 1, 1, 3])
    scale = tf.reshape(tf.constant(preprocessing_params.scale),
                       shape=[1, 1, 1, 3])
    x = (images[:, :, :, ::-1] - offset) / scale

    for layer in vgg_old.layers[1:]:
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.Conv2D):
            assert config['padding'] == 'same'
            config['padding'] = 'valid'
            x = ReflectionPadding2D(padding=1)(x)

        new_layer = layer.__class__.from_config(config)
        x = new_layer(x)

    vgg_model = tf.keras.Model(inputs=[images], outputs=[x])
    vgg_model.set_weights(vgg_old.get_weights())
    del vgg_old

    return vgg_model


class StyleTransferNetwork(tf.keras.Model):
    _ENCODING_LAYERS = [
        'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1'
    ]
    _MSE_LOSS = tf.losses.MeanSquaredError(reduction='none')

    def __init__(self, preprocessing_params, encoder_weights, **kwargs):
        super(StyleTransferNetwork, self).__init__(**kwargs)

        self.preprocessing_params = preprocessing_params
        self.encoder_weights = encoder_weights
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.adain = AdaptiveInstanceNormalization()

    def _build_encoder(self):
        base_model = get_vgg_model(self.preprocessing_params)
        base_model.load_weights(self.encoder_weights)

        logging.info('Initialized encoder with weights from {}'.format(
            self.encoder_weights))

        encoder = tf.keras.Model(
            inputs=base_model.inputs,
            outputs={
                layer: tf.cast(base_model.get_layer(layer).output,
                               dtype=tf.float32)
                for layer in StyleTransferNetwork._ENCODING_LAYERS
            },
            name='encoder')
        encoder.trainable = False
        return encoder

    def _build_decoder(self):
        conv2d_valid_padding = functools.partial(tf.keras.layers.Conv2D,
                                                 kernel_size=(3, 3),
                                                 strides=1,
                                                 padding='valid')
        nearest_upsampling_2x = functools.partial(tf.keras.layers.UpSampling2D,
                                                  size=2,
                                                  interpolation='nearest')

        inputs = tf.keras.Input(shape=[None, None, 512], name='decoder_input')
        x = inputs
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=256,
                                 activation='relu',
                                 name='decoder_block4_conv1')(x)
        x = nearest_upsampling_2x(name='decoder_block3_pool')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=256,
                                 activation='relu',
                                 name='decoder_block3_conv4')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=256,
                                 activation='relu',
                                 name='decoder_block3_conv3')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=256,
                                 activation='relu',
                                 name='decoder_block3_conv2')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=128,
                                 activation='relu',
                                 name='decoder_block3_conv1')(x)
        x = nearest_upsampling_2x(name='decoder_block2_pool')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=128,
                                 activation='relu',
                                 name='decoder_block2_conv2')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=64,
                                 activation='relu',
                                 name='decoder_block2_conv1')(x)
        x = nearest_upsampling_2x(name='decoder_block1_pool')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=64,
                                 activation='relu',
                                 name='decoder_block1_conv2')(x)
        x = ReflectionPadding2D(padding=1)(x)
        x = conv2d_valid_padding(filters=3,
                                 activation='linear',
                                 name='decoder_block1_conv1')(x)
        outputs = x
        decoder = tf.keras.Model(inputs=[inputs],
                                 outputs=[outputs],
                                 name='Decoder')
        return decoder

    def call(self, x, training=False):
        style_images, content_images, alpha = x
        alpha = tf.cast(alpha, dtype=tf.float32)

        # f(s) and f(c)
        style_features = self.encoder(style_images)
        content_features = self.encoder(content_images)

        # t = adain(f(s), f(c))
        normalized_features = self.adain(
            (style_features['block4_conv1'], content_features['block4_conv1']))
        normalized_features = tf.cast(normalized_features, dtype=tf.float32)
        normalized_features = \
            alpha * normalized_features + (1 - alpha) * content_features['block4_conv1']  # noqa: E501

        # T(c, s) = g(t)
        synthesized_images = self.decoder(normalized_features)
        synthesized_images = tf.cast(synthesized_images, dtype=tf.float32)

        if not training:
            return {'synthesized_images': synthesized_images}

        synthesized_features = self.encoder(synthesized_images)

        content_loss = StyleTransferNetwork._compute_content_loss(
            synthesized_features, normalized_features)

        style_loss = StyleTransferNetwork._compute_style_loss(
            synthesized_features, style_features)

        return {
            'synthesized_images': synthesized_images,
            'loss': {
                'content-loss': content_loss,
                'style-loss': style_loss
            }
        }

    def _compute_content_loss(synthesized_features, normalized_features):
        return tf.reduce_mean(StyleTransferNetwork._MSE_LOSS(
            synthesized_features['block4_conv1'], normalized_features))

    def _compute_style_loss(synthesized_features, style_features):
        style_loss = 0
        for level in StyleTransferNetwork._ENCODING_LAYERS:
            style_features_mean, style_features_variance = tf.nn.moments(
                style_features[level], axes=[1, 2])
            style_features_std = tf.sqrt(style_features_variance +
                                         tf.keras.backend.epsilon())

            synthesized_features_mean, synthesized_features_variance = \
                tf.nn.moments(synthesized_features[level], axes=[1, 2])
            synthesized_features_std = tf.sqrt(synthesized_features_variance +
                                               tf.keras.backend.epsilon())

            loss_mu = tf.reduce_mean(StyleTransferNetwork._MSE_LOSS(
                synthesized_features_mean, style_features_mean))
            loss_sigma = tf.reduce_mean(StyleTransferNetwork._MSE_LOSS(
                synthesized_features_std, style_features_std))
            style_loss += (loss_mu + loss_sigma)
        return style_loss
