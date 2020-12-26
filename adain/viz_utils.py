import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def renormalize(image):
    image = (image + [103.939, 116.779, 123.68])[:, :, ::-1]
    return np.uint8(np.clip(image, 0, 255))


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def prepare_visualization_image(style_image, content_image, generated_image):
    fig, ax = plt.subplots(1, 3, figsize=(18, 8))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('style image')
    plt.imshow(renormalize(style_image))
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('content image')
    plt.imshow(renormalize(content_image))
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('generated image')
    plt.imshow(renormalize(generated_image))
    return plot_to_image(fig)
