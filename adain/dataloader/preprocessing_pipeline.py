import tensorflow as tf


class PreprocessingPipeline:

    def __init__(self, params):
        self.input_shape = params.input.input_shape
        self.preprocessing_params = params.dataloader_params.preprocessing
        self.augmentation_params = params.dataloader_params.augmentations

    def _resize_and_crop(self, image):
        image_size = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
        ratio = self.augmentation_params.min_side / tf.minimum(
            image_size[0], image_size[1])
        new_image_size = tf.cast(image_size * ratio, dtype=tf.int32)

        image = tf.image.resize(image, size=new_image_size)
        image = tf.image.random_crop(
            image, size=[self.input_shape[0], self.input_shape[1], 3])

        return image, image_size, new_image_size, ratio

    def __call__(self, sample, return_labels=False):
        if self.augmentation_params.horizontal_flip:
            image = tf.image.random_flip_left_right(sample["image"])

        image, image_size, new_image_size, ratio = self._resize_and_crop(image)

        offset = tf.reshape(tf.constant(self.preprocessing_params.offset),
                            shape=[1, 1, 3])
        scale = tf.reshape(tf.constant(self.preprocessing_params.scale),
                           shape=[1, 1, 3])
        image -= offset
        image /= scale
        return {
            'image': image,
            'image_size': image_size,
            'new_image_size': new_image_size,
            'ratio': ratio
        }
