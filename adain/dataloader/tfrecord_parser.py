import tensorflow as tf


def parse_example(example_proto):
    parsed_example = tf.io.parse_single_example(
        example_proto,
        {
            'image': tf.io.FixedLenFeature([], tf.string)
        })

    image = tf.io.decode_image(parsed_example['image'], channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image.set_shape([None, None, 3])

    return {
        'image': image
    }
