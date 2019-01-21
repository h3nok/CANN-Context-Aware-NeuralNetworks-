import atexit
import math
import os
import time
from time import clock

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.image as tfimage
from keras.backend import tensorflow_backend as KTF
from PIL import Image
from sklearn.feature_extraction import image
from sklearn.feature_extraction.image import extract_patches

from cc_utils import ImagePlot as IMPLOT
from reconstructor import reconstruct_from_patches
from timer import endlog, log
import map_measure


def _patchify_tf(image_data, ksize_rows, ksize_cols, strides_rows, strides_cols, rates=[1, 1, 1, 1], padding='VALID'):
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
                patch_width, patch_height, 3])

            list_of_patches.append(sess.run(patch))

    return list_of_patches


def generate_patches_v1(sess, image_data_byes, ksize_rows, ksize_cols):
    """Split image into patches usinf tensorflow 
    Arguments:
        sess {tf.Session} -- tensorflow session to evaluate tensors 
        image_data_byes {bytes} -- decode_image into tensorflow tensor 
        ksize_rows {pixels} -- patch height  
        ksize_cols {pixels} -- patch width 
    """
    start = clock()
    atexit.register(endlog, start)
    log("Start Program - func: generate_patches(...)")

    # + the centers of two consecutive patches.
    strides_rows = ksize_rows  # 128
    strides_cols = ksize_cols  # 128

    image = tf.image.decode_image(image_string, channels=3)

    image_patches = _patchify_tf(
        image, ksize_rows,
        ksize_cols, strides_rows,
        strides_cols)

    p = sess.run(tf.shape(image_patches))
    number_patch_row = p[1]
    number_patch_col = p[2]

    return image_patches, number_patch_row, number_patch_col


def generate_patches_v2(image_string, input_h, input_w, patch_h, patch_w):
    """
    Splits an image into patches of size patch_h x patch_w
    Input: image of shape [image_h, image_w, image_ch]
    Output: batch of patches shape [n, patch_h, patch_w, image_ch]

    Arguments:
        image_string {sample} -- input string 
        input_h {pixels} -- input height in pixels 
        input_w {pixels} -- input width in pixels 
        patch_h {pixels} -- patch height to extract 
        patch_w {pixels} -- patch width to extract 
    """

    image = tf.image.decode_image(image_string, channels=3)

    image = tf.reshape(image, [input_h, input_w, 3])

    assert image.shape.ndims == 3

    pad = [[0, 0], [0, 0]]
    image_h = image.shape[0].value
    image_w = image.shape[1].value
    image_ch = image.shape[2].value
    p_area = patch_h * patch_w

    assert image_h == image_w and patch_h == patch_w, "Traning sample and patches must be of square size!"

    patches = tf.space_to_batch_nd([image], [patch_h, patch_w], pad)
    patches = tf.split(patches, p_area, 0)
    patches = tf.stack(patches, 3)
    patches = tf.reshape(patches, [-1, patch_h, patch_w, image_ch])

    return patches, image


def image_data_conserved(original, reconstructed):
    return tf.reduce_all(tf.math.equal(original, reconstructed))


if __name__ == '__main__':
    image_file = 'husky.jpg'
    patch_width = 32
    patch_height = 32
    channel = 3
    input_size = (224, 224)
    image_patches = None

    assert patch_height == patch_width, "CC doesn't support different sized patches!"

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)) as sess:

        image_string = tf.gfile.FastGFile(image_file, 'rb').read()
        patches, original = generate_patches_v2(
            image_string, input_size[0], input_size[1], patch_width, patch_height)

        reconstructed = reconstruct_from_patches(
            patches, input_size[0], input_size[1])

        data_conserved = sess.run(
            image_data_conserved(original, reconstructed))

        if not data_conserved:
            print("Reconstruction data loss, skipping sample")
