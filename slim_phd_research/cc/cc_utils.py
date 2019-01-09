import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
import numpy as np

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def read_jpeg_to_ndarray_using_cv(self, image_file):
        image_data = cv2.imread(image_file)
        b,g,r = cv2.split(image_data)
        assert len(image_data.shape) == 3
        assert image_data.shape[2] == 3
        (h, w) = image_data.shape[:2]
        assert h == w

        return image_data, r,g,b

    def open(self, image_file,resize=(32,32)):
        img = mpimg.imread(image_file)

        return img

class ImagePlot(object):

    def plot_multiple(self,tensor_list, subsample):
        rows = subsample/2
        cols = subsample/2

        fig = plt.figure(figsize=(8,8))

        for i in range(1,subsample+1):
            fig.add_subplot(rows,cols,i)
            for j in range(1,subsample+1):
                print (i)
                print (j)
                plt.imshow(tensor_list[i-1,j-1])

        plt.show()
    
    def plot_patches(self,patches, fignum=None, low=0, high=0):
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



