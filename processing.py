import os
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import requests
from io import BytesIO


class ImageProcessing:
    def __init__(self):
        hub_handle = "https://tfhub.dev/google/" \
                     "magenta/arbitrary-image-stylization-v1-256/2"
        self.hub_module = hub.load(hub_handle)

    @staticmethod
    def _get_image(url):
        if url.lower().startswith('http'):
            r = requests.get(url)
            url = BytesIO(r.content)
        image = tf.io.decode_image(
            tf.io.read_file(url),
            channels=3,
            dtype=tf.float32)[tf.newaxis, ...]
        return image

    @staticmethod
    def __crop_center(image, image_size=(256, 256)):
        shape = image.shape
        new_shape = min(shape[1], shape[2])
        offset_y = max(shape[1] - shape[2], 0) // 2
        offset_x = max(shape[2] - shape[1], 0) // 2
        image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x,
            new_shape, new_shape
        )
        image = tf.image.resize(image, image_size, preserve_aspect_ratio=True)
        return image

    def _forward(self, content_image, style_image):
        outputs = self.hub_module(
            tf.constant(content_image),
            tf.constant(style_image)
        )
        stylized_image = outputs[0]
        return np.squeeze(stylized_image)

    def __call__(self, content_url, style_url):
        content = self._get_image(content_url)
        style = self._get_image(style_url)
        content = self.__crop_center(content)
        style = self.__crop_center(style)
        stylized_image = self._forward(content, style)
        return stylized_image

    @staticmethod
    def show(image):
        plt.imshow(image)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.show()


