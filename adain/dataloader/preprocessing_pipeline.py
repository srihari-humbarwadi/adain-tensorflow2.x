import tensorflow as tf


class PreprocessingPipeline:

    def __init__(self, params, is_validation_dataset):
        self.input_shape = params.input.input_shape
        self.preprocessing_params = params.dataloader_params.preprocessing
        self.augmentation_params = params.dataloader_params.augmentations
        self.is_validation_dataset = is_validation_dataset

    def _resize_and_crop(self, image):
        image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        ratio = self.augmentation_params.min_side / tf.minimum(
            image_size[0], image_size[1])
        new_image_size = tf.cast(image_size * ratio, dtype=tf.int32)

        image = tf.image.resize(image, size=new_image_size)
        image = tf.image.random_crop(
            image, size=[self.input_shape[0], self.input_shape[1], 3])

        return image, image_size, new_image_size, ratio

    def _resize(self, image):
        image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        max_side = tf.maximum(image_size[0], image_size[1])

        if max_side > self.augmentation_params.max_side:
            ratio = self.augmentation_params.max_side / max_side
            new_image_size = tf.cast(image_size * ratio, dtype=tf.int32)
            image = tf.image.resize(image, size=new_image_size)

        else:
            ratio = 1.0
            new_image_size = tf.cast(image_size, dtype=tf.int32)

        return image, image_size, new_image_size, ratio

    def inference_pipeline(self, images):
        image, image_size, new_image_size, ratio = self._resize(images[0])
        offset = tf.reshape(tf.constant(self.preprocessing_params.offset),
                            shape=[1, 1, 3])
        scale = tf.reshape(tf.constant(self.preprocessing_params.scale),
                           shape=[1, 1, 3])

        if self.preprocessing_params.use_bgr:
            image = image[:, :, ::-1]

        image = tf.math.divide_no_nan(image - offset, scale)
        return tf.expand_dims(image, axis=0)

    def denormalize(self, images):
        image = images[0]
        offset = tf.reshape(tf.constant(self.preprocessing_params.offset),
                            shape=[1, 1, 3])
        scale = tf.reshape(tf.constant(self.preprocessing_params.scale),
                           shape=[1, 1, 3])

        image = image * scale + offset

        if self.preprocessing_params.use_bgr:
            image = image[:, :, ::-1]

        image = tf.clip_by_value(image, 0, 255)

        return tf.cast(tf.expand_dims(image, axis=0), dtype=tf.uint8)

    def __call__(self, sample, return_labels=False):
        image = sample["image"]

        if not self.is_validation_dataset:
            image, image_size, new_image_size, ratio = \
                self._resize_and_crop(image)
        else:
            image, image_size, new_image_size, ratio = \
                self._resize(image)

        if self.augmentation_params.horizontal_flip and (not self.is_validation_dataset):  # noqa: E501
            image = tf.image.random_flip_left_right(image)

        if self.preprocessing_params.use_bgr:
            image = image[:, :, ::-1]

        offset = tf.reshape(tf.constant(self.preprocessing_params.offset),
                            shape=[1, 1, 3])
        scale = tf.reshape(tf.constant(self.preprocessing_params.scale),
                           shape=[1, 1, 3])

        image = tf.math.divide_no_nan(image - offset, scale)

        return {
            'image': image,
            'image_size': image_size,
            'new_image_size': new_image_size,
            'ratio': ratio
        }
