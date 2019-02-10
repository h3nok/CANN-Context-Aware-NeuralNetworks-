from numpy.linalg import norm
import tensorflow as tf
import numpy as np


def L2Norm(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # combine the two tensors into one
    patch_data = np.concatenate((patch_1, patch_2)).flatten()
    return round(norm(patch_data, 2), 4)
