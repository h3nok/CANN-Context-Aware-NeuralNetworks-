import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np
import sys


class ImageHelper(object):

    def plot_patches(self, patches, fignum=None, low=0, high=0):
        """
        Given a stack of 2D patches indexed by the first dimension, plot the
        patches in subplots. 

        'low' and 'high' are optional arguments to control which patches
        actually get plotted. 'fignum' chooses the figure to plot in.
        """
        try:
            istate = plt.isinteractive()
            plt.ioff()
            if fignum is None:
                fig = plt.gcf()
            else:
                fig = plt.figure(fignum)
            if high == 0:
                high = len(patches)
            pmin, pmax = patches.min(), patches.max()
            dims = np.ceil(np.sqrt(high - low))
            for idx in range(high - low):
                spl = plt.subplot(dims, dims, idx + 1)
                ax = plt.axis('off')
                im = plt.imshow(patches[idx], cmap=matplotlib.cm.gray)
                cl = plt.clim(pmin, pmax)
            plt.show()
        finally:
            plt.interactive(istate)

    def plot_patches_tf(image_patches, sess, ksize_rows, ksize_cols):
        #x = sess.run(image_patches)
        #nr = x.shape[1]
        #nc = x.shape[2]
        #del x
        a = sess.run(tf.shape(image_patches))
        nr, nc = a[1], a[2]
        print('\nwidth: {}; height: {}'.format(nr, nc), file=sys.stderr)

        # figsize: width and height in inches. can be changed to make
        # +output figure fit well. The default often works well.

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
                plt.imshow(sess.run(patch))
                print('processed {},{} patch, {}.'.format(
                    i, j, i*nc+j), file=sys.stderr)
        return fig

    def show_images(self, images, cols=1, titles=None):
        """Display a list of images in a single figure with matplotlib.

        Parameters
        ---------
        images: List of np.arrays compatible with plt.imshow.

        cols (Default = 1): Number of columns in figure (number of rows is 
                            set to np.ceil(n_images/float(cols))).

        titles: List of titles corresponding to each image. Must have
                the same length as titles.
        """
        assert((titles is None)or (len(images) == len(titles)))
        plt.tight_layout()
        n_images = len(images)
        if titles is None:
            titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
            if image.ndim == 2:
                plt.gray()
            plt.imshow(image)
            a.set_title(title)
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

        return fig
