import tensorflow as tf
from skimage.measure import compare_ssim as ssim
import numpy as np


def SSIM(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    value = ssim(patch_1, patch_2)
    return round(value, 4)
