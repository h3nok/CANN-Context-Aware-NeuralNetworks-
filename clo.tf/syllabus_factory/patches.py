import numpy as np
from itertools import product

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.util.shape import view_as_windows


def plot_image_patches(x, ksize_rows=299, ksize_cols=299):
    nr = x.shape[1]
    nc = x.shape[2]
    # figsize: width and height in inches. can be changed to make
    # +output figure fit well.
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


class ImagePatches(object):
    def __init__(self):
        self._size = tf.placeholder(dtype=tf.int)


def patchify(patches: np.ndarray, patch_size, step: int = 1):
    return view_as_windows(patches, patch_size, step)


def unpatchify(patches: np.ndarray, imsize):

    assert len(patches.shape) >= 4

    i_h, i_w, channels = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, _, p_h, p_w, _ = patches.shape

    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i, j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor
