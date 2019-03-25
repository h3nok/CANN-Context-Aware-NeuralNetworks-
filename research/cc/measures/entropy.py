try:
    from cc.ITT import itt as itt
except(Exception, ImportError) as error:
    print(error)
    from ITT import itt as itt
import numpy as np
import tensorflow as tf


def entropy(patch):
    # flatten the tensor into a sigle dimensinoal array
    patch_data = patch.flatten()
    e = itt.entropy(patch_data)

    return round(float(e), 4)  # result x.xxxx


def cross_entropy(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # flatten the tensor
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


def conditional_entropy(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # flatten the tensor
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()
    ce = itt.entropy_conditional(patch_1, patch_2)
    return round(float(ce), 4)  # result x.xxxx


def residual_entropy(patch_1):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."

    patch_1 = patch_1.flatten()

    re = round(itt.entropy_residual(patch_1), 4)  # result x.xxxx

    return re


def joint_entropy_tf(patch_1, patch_2):

    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert patch_1 != patch_2, "Patches are binary equivalent, Distance = 0"

    sess = tf.get_default_session()

    # combine the two tensors into one
    patch_data = sess.run(
        tf.concat([patch_1,
                   patch_2], 0))
    # flatten the tensor
    patch_data = sess.run(tf.reshape(patch_data, [-1]))

    je = round(itt.entropy_joint(patch_data), 4)  # result x.xxxx

    return je


def joint_entropy(patch_1, patch_2):

    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # combine the two tensors into one
    patch_data = np.concatenate((patch_1, patch_2)).flatten()
    je = round(itt.entropy_joint(patch_data), 4)  # result x.xxxx

    return je


def kl_divergence(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # combine the two tensors into one
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    je = round(itt.divergence_kullbackleibler(patch_1, patch_2), 4)  # result x.xxxx

    return je
