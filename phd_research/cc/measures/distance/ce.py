
import numpy as np
import tensorflow as tf
try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    print (error)
    from ITT import itt as itt


def ConditionalEntropy(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # flatten the tensor into a sigle dimensinoal array
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()
    ce = itt.entropy_conditional(patch_1, patch_2)
    return round(float(ce), 4)  # result x.xxxx
