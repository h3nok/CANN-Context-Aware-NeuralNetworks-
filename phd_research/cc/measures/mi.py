import numpy as np
import tensorflow as tf
try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    print (error)
    from ITT import itt as itt


def MutualInformationTF(patch_1, patch_2):

    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert patch_1 != patch_2, "Patches are binary equivalent, Distance = 0"

    sess = tf.get_default_session()

    # combine the two tensors into one
    # flatten the tensor into a sigle dimensinoal array
    patch_1 = sess.run(tf.reshape(patch_1, [-1]))
    patch_2 = sess.run(tf.reshape(patch_2, [-1]))

    mi = round(itt.information_mutual(patch_1, patch_2), 4)  # result x.xxxx

    return mi


def MutualInformation(patch_1, patch_2):

    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # combine the two tensors into one
    # flatten the tensor into a sigle dimensinoal array
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    mi = round(itt.information_mutual(patch_1, patch_2), 4)  # result x.xxxx

    return mi
