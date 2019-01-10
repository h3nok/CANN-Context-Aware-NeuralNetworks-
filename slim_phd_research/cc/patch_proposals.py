import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.image as tfimage
from keras.backend import tensorflow_backend as KTF
from PIL import Image
from skimage.util.shape import view_as_blocks, view_as_windows
from sklearn.feature_extraction import image
from sklearn.feature_extraction.image import extract_patches

from cc_utils import ImagePlot as IMPLOT
from cc_utils import ImageReader as CCIR
import patches


def generate_patches(image_tensor, patch_size):
    if not isinstance(patch_size, tuple):
        raise ValueError("Patch size must be of tuple type, (w,h)")

    patches = view_as_blocks(image_tensor, patch_size)

    # collapse the last two dimensions in one
    patches = patches.reshape(patches.shape[0], patches.shape[1], -1)

    return patches, len(patches)


def generate_image_patches_tf(image_data, ksize_rows, ksize_cols, strides_rows, strides_cols, rates=[1, 1, 1, 1], padding='SAME'):
    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1]

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]

    image = tf.expand_dims(image_data, 0)
    image_patches = tf.extract_image_patches(
        image, ksizes, strides, rates, padding)

    return image_patches


def generate_patches_test():
    image_file = 'husky.jpg'
    image_string = tf.gfile.FastGFile(image_file, 'rb').read()
    ksize_rows = 64
    ksize_cols = 64

    # strides_rows and strides_cols determine the distance between
    #+ the centers of two consecutive patches.
    strides_rows = ksize_rows  # 128
    strides_cols = ksize_cols  # 128

    image = tf.image.decode_image(image_string, channels=3)

    image_patches = generate_image_patches_tf(
        image, ksize_rows, ksize_cols, strides_rows, strides_cols)

    sess = tf.InteractiveSession()

    # Method 2:
    fig = patches.plot_image_patches2(
        image_patches, sess, ksize_rows, ksize_cols)

    # plt.savefig('image_patches.png', bbox_inches='tight',dpi=300) # use dpi to control image size, e.g. 800
    # use dpi to control image size, e.g. 800
    plt.savefig('image_patches.png', bbox_inches='tight', dpi=120)
    plt.show()

    plt.close(fig)

    sess.close()


if __name__ == '__main__':
    generate_patches_test()
