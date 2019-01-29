import numpy as np
from pyitlib import discrete_random_variable as drv
import tensorflow as tf


def KullbackLeiblerDivergence(patch_1, patch_2):

    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # combine the two tensors into one
    # and flatten the tensor into a sigle dimensinoal array
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    je = round(drv.divergence_kullbackleibler(patch_1, patch_2), 4)  # result x.xxxx
    return je
