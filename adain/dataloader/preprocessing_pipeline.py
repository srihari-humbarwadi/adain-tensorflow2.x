import tensorflow as tf


class PreprocessingPipeline:

    def __init__(self, params, is_validation_dataset):
        self.input_shape = params.input.input_shape
        self.augmentation_params = params.dataloader_params.augmentations
        self.is_validation_dataset = is_validation_dataset

    def _resize_and_crop(self, image):
        image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        ratio = self.augmentation_params.min_side / tf.minimum(
            image_size[0], image_size[1])
        new_image_size = tf.cast(tf.round(image_size * ratio), dtype=tf.int32)

        image = tf.image.resize(image, size=new_image_size)
        image = tf.image.random_crop(
            image, size=[self.input_shape[0], self.input_shape[1], 3])

        return image, image_size, new_image_size, ratio

    def _resize(self, image):
        image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        ratio = self.augmentation_params.min_side / tf.minimum(
            image_size[0], image_size[1])
        new_image_size = tf.cast(tf.round(image_size * ratio), dtype=tf.int32)
        image = tf.image.resize(image, size=new_image_size)
        return image, image_size, new_image_size, ratio

    def inference_pipeline(self, images):
        image, image_size, new_image_size, ratio = self._resize(images[0])
        return tf.expand_dims(image, axis=0)

    def denormalize(self, images):
        pixel_mins = tf.reduce_min(images, axis=[1, 2, 3])
        pixel_maxs = tf.reduce_max(images, axis=[1, 2, 3])
        pixel_range = pixel_maxs - pixel_mins

        images = 255.0 * (images - pixel_mins) / pixel_range
        images = tf.clip_by_value(images, 0, 255)

        return tf.cast(images, dtype=tf.uint8)

    def __call__(self, sample):
        image = sample["image"]

        if not self.is_validation_dataset:
            image, image_size, new_image_size, ratio = \
                self._resize_and_crop(image)
        else:
            image, image_size, new_image_size, ratio = \
                self._resize(image)

        if self.augmentation_params.horizontal_flip and (not self.is_validation_dataset):  # noqa: E501
            image = tf.image.random_flip_left_right(image)

        return {
            'image': image,
            'image_size': image_size,
            'new_image_size': new_image_size,
            'ratio': ratio
        }
