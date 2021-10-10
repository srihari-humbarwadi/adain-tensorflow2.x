import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def read_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes)
    image.set_shape([None, None, None])

    if image.get_shape().as_list()[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    image = tf.expand_dims(image, axis=0)
    return tf.cast(image, dtype=tf.float32)


def read_image_cv2(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image[None, :, :, :]
    return image


def imshow(image, figsize=(16, 9), title=None):
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(np.uint8(image))

    if title:
        plt.title(title)


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    return image.numpy()


def prepare_visualization_image(content_image, style_image,
                                stylized_image, figsize=(18, 8)):
    style_image = np.uint8(style_image)
    content_image = np.uint8(content_image)
    stylized_image = np.uint8(stylized_image)

    fig, ax = plt.subplots(1, 3, figsize=figsize)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('content image')
    plt.imshow(content_image)
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('style image')
    plt.imshow(style_image)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('stylized image')
    fig.tight_layout()
    plt.imshow(stylized_image)
    return plot_to_image(fig)
