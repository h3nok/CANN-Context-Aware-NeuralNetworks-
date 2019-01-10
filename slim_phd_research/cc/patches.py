import os
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import matplotlib.gridspec as gridspec
from skimage.util.shape import view_as_blocks, view_as_windows
from itertools import product
import gc


class ImagePatches(object):
    def __init__(self):
        self._size = tf.placeholder(dtype=tf.int)

    def plot_image_patches(self, x, ksize_rows=299, ksize_cols=299):
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


def patchify(patches: np.ndarray, patch_size, step: int = 1):
    print(patches.shape)
    return view_as_windows(patches, patch_size, step)


def unpatchify(patches: np.ndarray, imsize):

    assert len(patches.shape) >= 4

    i_h, i_w, channels = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, _, p_h, p_w, _ = patches.shape

    # Calculat the overlap size in each axis
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


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    wind = None
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    # else:
    #     wind = _spline_window(window_size, power)
    #     wind = np.expand_dims(np.expand_dims(wind, 3), 3)
    #     wind = wind * wind.transpose(1, 0, 2)
    #     if PLOT_PROGRESS:
    #         # For demo purpose, let's look once at the window:
    #         plt.imshow(wind[:, :, 0], cmap="viridis")
    #         plt.title("2D Windowing Function for a Smooth Blending of "
    #                   "Overlapping Patches")
    #         plt.show()
    #     cached_2d_windows[key] = wind

    return wind


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Create tiled overlapping patches.
    Returns:
        5D numpy array of shape = (
            nb_patches_along_X,
            nb_patches_along_Y,
            patches_resolution_along_X,
            patches_resolution_along_Y,
            nb_output_channels
        )
    Note:
        patches_resolution_along_X == patches_resolution_along_Y == window_size
    """
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []

    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, padx_len-window_size+1, step):
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)

    # Here, `gc.collect()` clears RAM between operations.
    # It should run faster if they are removed, if enough memory is available.
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()

    # subdivs = pred_func(subdivs)
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()

    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()

    return subdivs


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    """
    Merge tiled overlapping patches smoothly.
    """
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]

    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, padx_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i +
                                                    window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1

    return y / (subdivisions ** 2)
