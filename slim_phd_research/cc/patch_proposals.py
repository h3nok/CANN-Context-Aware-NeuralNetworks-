import os
import atexit
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.image as tfimage
from keras.backend import tensorflow_backend as KTF
from PIL import Image
from sklearn.feature_extraction import image
import time
from sklearn.feature_extraction.image import extract_patches

from cc_utils import ImagePlot as IMPLOT
from cc_utils import ImageReader as CCIR
from time import clock
from timer import endlog, log
from PIL import Image
from patches import patchify, unpatchify, _windowed_subdivs, _recreate_from_subdivs


def extract_patches(image_data, ksize_rows, ksize_cols, strides_rows, strides_cols, rates=[1, 1, 1, 1], padding='SAME'):
    """[summary]

    Arguments:
        image_data {tftensor} -- Read image into byes the follow by tf.image.decode_image
        ksize_rows {pixels} -- height of sliding windows  
        ksize_cols {pixels} -- width of sliding window 
        strides_rows {pixels} -- vertical distance between centers of two consecutive patches 
        strides_cols {pixels} -- horizontal distance between centers of two consecutive patches 

    Keyword Arguments:
        rates {list} -- [description] (default: {[1, 1, 1, 1]})
        padding {string} -- [description] (default: {'SAME'})
    """
    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1]

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]

    image = tf.expand_dims(image_data, 0)
    image_patches = tf.extract_image_patches(
        image, ksizes, strides, rates, padding)

    return image_patches


def _to_list(sess, tensor_patches, nr, nc):
    # convert to tf tensor to numpy array and return a list of patche
    list_of_patches = []
    for i in range(nr):
        for j in range(nc):
            patch = tf.reshape(tensor_patches[0, i, j, ], [
                ksize_rows, ksize_cols, 3])

            list_of_patches.append(sess.run(patch))

    return list_of_patches


def generate_patches(sess, image_data_byes, ksize_rows, ksize_cols):
    start = clock()
    atexit.register(endlog, start)
    log("Start Program - func: generate_patches(...)")

    # + the centers of two consecutive patches.
    strides_rows = ksize_rows  # 128
    strides_cols = ksize_cols  # 128

    image = tf.image.decode_image(image_string, channels=3)

    image_patches = extract_patches(
        image, ksize_rows, ksize_cols, strides_rows, strides_cols)

    p = sess.run(tf.shape(image_patches))
    number_patch_row = p[1]
    number_patch_col = p[2]

    return image_patches, number_patch_row, number_patch_col


def reconstruct(sess, image_patches, row, col, height, width, channels=3):
    start = clock()
    atexit.register(endlog, start)
    log("Start Program - func: reconstruct(...)")

    patches = _to_list(sess, image_patches, row, col)

    cx = 1
    cy = 0
    number_of_patches = len(patches)
    image_size = math.sqrt(number_of_patches)

    patch_w = patches[0].shape[0]
    patch_h = patches[0].shape[1]

    new_h = int(patch_h*image_size)
    new_w = int(patch_w*image_size)
    result_image = Image.new('RGB', (new_w, new_h))

    for patch in patches:
        print(patch.size)
        print(type(patch))
        result_image.paste(im=patch, box=(cx*patch_w, cy*patch_h))
        cx += 1
        if cx == width:
            cy += 1

    return result_image


def verify(original, reconstructed):
    pass


if __name__ == '__main__':

    image_file = 'husky.jpg'
    image_string = tf.gfile.FastGFile(image_file, 'rb').read()
    im = np.array(Image.open(image_file))
    ksize_rows = 64
    ksize_cols = 64
    image_patches = None
    nr = 0
    nc = 0


    sdfsdf
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:
        image_patches, nr, nc = generate_patches(
            sess, image_string, ksize_rows, ksize_cols)

        # fig = IMPLOT.plot_patches_tf(
        #     image_patches, sess, ksize_rows, ksize_cols)
        # # plt.savefig('image_patches.png', bbox_inches='tight', dpi=120)
        # # plt.show()
        # # plt.close(fig)
        # # plt.close()
        reconstructed = reconstruct(sess, image_patches, nr, nc, 224, 224)
