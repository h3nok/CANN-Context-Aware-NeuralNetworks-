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


def extract_patches_tf():
    patch_width = 8
    patch_height = 8

    reader = tf.WholeFileReader()

    input_sample = tf.train.string_input_producer(
        ["C:\\phd\\Samples\\resized.png"])

    key, value = reader.read(input_sample)
    image = tfimage.decode_png(value)

    image_file = "C:\\phd\\Samples\\resized.jpg"
    image_string = tf.gfile.FastGFile(image_file, 'rb').read()

    ksize_rows = 32
    ksize_cols = 32
    strides_rows = 32
    strides_cols = 32

    sess = tf.InteractiveSession()

    image = tf.image.decode_image(image_string, channels=3)

    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1]

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]

    # The document is unclear. However, an intuitive example posted on StackOverflow illustrate its behaviour clearly.
    # http://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
    rates = [1, 1, 1, 1]  # sample pixel consecutively

    # padding algorithm to used
    padding = 'SAME'  # or 'SAME'
    image = tf.expand_dims(image, 0)
    image_patches = tf.extract_image_patches(
        image, ksizes, strides, rates, padding)

    # print image shape of image patche
    print(sess.run(tf.shape(image_patches)))

    # image_patches is 4 dimension array, you can use tf.squeeze to squeeze it, e.g.
    # image_patches = tf.squeeze(image_patche)
    # retrieve the 1st patch
    patch_count = 0

    for i in range(0, 7):
        for j in range(0, 7):
            patch1 = image_patches[0, i, j, ]

            print(image_patches)
            # reshape
            patch1 = tf.reshape(patch1, [ksize_rows, ksize_cols, 3])

            # visualize image
            import matplotlib.pyplot as plt

            plt.imshow(sess.run(patch1))
            plt.show()
            patch_count += 1

    print("Number of patches: {}".format(patch_count))
    # close session
    sess.close()


def generate_patches(image_tensor, patch_size):
    if not isinstance(patch_size, tuple):
        raise ValueError("Patch size must be of tuple type, (w,h)")

    patches = view_as_blocks(image_tensor, patch_size)

    # collapse the last two dimensions in one
    patches = patches.reshape(patches.shape[0],patches.shape[1],-1)

    return patches, len(patches)

def extract_patches2(image_tensor, patch_size):
    if not isinstance(patch_size, tuple):
        raise ValueError("Patch size must be of tuple type, (w,h)")
    assert len(image_tensor.shape) == 3, "Supplied image is not color image"
    assert image_tensor.shape[0] == image_tensor.shape[1], "Sample is not sqaure, migh need resizing"
    max_patches = int(image_tensor.shape[0]/patch_size[0])
    patches = image.extract_patches_2d(image_tensor, patch_size)

    return patches, len(patches)

def test_generate_patches():
    image_file = "C:\\phd\\Samples\\resized.png"
    if not os.path.exists(image_file):
        print("ERROR: file \'{}\' not found ".format(image_file))
    image_reader = CCIR()
    image_plot = IMPLOT()
    image_tensor, r, g, b = image_reader.read_jpeg_to_ndarray_using_cv(image_file)
    patches,  number_of_patches = generate_patches(image_tensor, (4, 4,3))

    print("Patch shape: {}".format(patches.shape))
    print("Number of patches: {} ".format(number_of_patches))



if __name__ == '__main__':
    test_generate_patches()
