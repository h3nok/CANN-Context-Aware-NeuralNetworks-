import numpy as np

try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    from ITT import itt as itt


def cross_entropy(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # flatten the tensor into a sigle dimensinoal array
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()
    ce = itt.entropy_cross(patch_1, patch_2)
    return round(float(ce), 4)  # result x.xxxx


def cross_entropy_pmf(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # flatten the tensor into a single dimensional array
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()
    ce = itt.entropy_cross_pmf(patch_1, patch_2)
    return round(float(ce), 4)  # result x.xxxx
