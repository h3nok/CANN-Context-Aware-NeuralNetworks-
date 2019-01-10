import os
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf 
import sys
import matplotlib.gridspec as gridspec

class ImagePatches(object):
    def __init__(self):
        self._size = tf.placeholder(dtype=tf.int)

def plot_image_patches(x, ksize_rows=299, ksize_cols=299):
    nr = x.shape[1]
    nc = x.shape[2]
    # figsize: width and height in inches. can be changed to make
    #+output figure fit well.
    #fig = plt.figure(figsize=(nr, nc))
    fig = plt.figure()
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.01, hspace=0.01)

    for i in range(nr):
        for j in range(nc):
            ax = plt.subplot(gs[i*nc+j])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('auto')
            plt.imshow(x[0, i, j, ].reshape(ksize_rows, ksize_cols, 3))
    return fig

def plot_image_patches2(image_patches, sess, ksize_rows, ksize_cols):
    #x = sess.run(image_patches)
    #nr = x.shape[1]
    #nc = x.shape[2]
    #del x
    a = sess.run(tf.shape(image_patches))
    nr, nc = a[1], a[2]
    print('\nwidth: {}; height: {}'.format(nr, nc), file=sys.stderr)
    # figsize: width and height in inches. can be changed to make
    #+output figure fit well. The default often works well.
    #fig = plt.figure(figsize=(nr, nc))
    fig = plt.figure()
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.01, hspace=0.01)

    for i in range(nr):
        for j in range(nc):
            ax = plt.subplot(gs[i*nc+j])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('auto')
            patch = tf.reshape(image_patches[0, i, j, ], [
                               ksize_rows, ksize_cols, 3])
            #patch = tf.image.random_brightness(patch, 0.3)
            #patch = tf.image.random_contrast(patch, 0.1, 0.9)
            #patch = tf.image.random_saturation(patch, 0.1, 0.9)
            #patch = tf.image.random_hue(patch, 0.4)
            #patch = tf.image.random_flip_up_down(patch, 0.4)
            plt.imshow(sess.run(patch))
            print('processed {},{} patch, {}.'.format(
                i, j, i*nc+j), file=sys.stderr)
    return fig
