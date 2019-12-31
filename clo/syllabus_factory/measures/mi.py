import numpy as np
import tensorflow as tf
try:
    from ITT import itt as itt
except(Exception, ImportError) as error:
    print (error)
    from ITT import itt as itt


def mutual_information_tf(patch_1, patch_2):

    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert patch_1 != patch_2, "Patches are binary equivalent, Distance = 0"

    sess = tf.get_default_session()

    # combine,flatten the two tensors into one
    patch_1 = sess.run(tf.reshape(patch_1, [-1]))
    patch_2 = sess.run(tf.reshape(patch_2, [-1]))

    mi = round(itt.information_mutual(patch_1, patch_2), 4)  # result x.xxxx

    return mi


def mutual_information(patch_1, patch_2):

    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    # assert not np.array_equiv(
    #     patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # combine the two tensors into one
    # flatten the tensor into a single dimensional array
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    mi = round(itt.information_mutual(patch_1, patch_2), 4)  # result x.xxxx

    return mi


def normalized_mutual_information(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    nmi = round(itt.information_mutual_normalised(patch_1, patch_2), 4)  # result x.xxxx

    return nmi


def enigmatic_information(patch_1):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."

    patch_1 = patch_1.flatten()

    ei = round(itt.information_enigmatic(patch_1), 4)  # result x.xxxx

    return ei


def lautum_information(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    li = round(itt.information_lautum(patch_1, patch_2), 4)  # result x.xxxx

    return li


def multi_information(patch_1):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."

    patch_1 = patch_1.flatten()

    ei = round(itt.information_multi(patch_1), 4)  # result x.xxxx

    return ei


def exogenous_local_information(patch):
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a single dimensional array
    patch = patch.flatten()
    eli = itt.information_exogenous_local(patch)

    return round(float(eli), 4)  # result x.xxxx


def information_interaction(patch):
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a single dimensional array
    patch = patch.flatten()
    eli = itt.information_interaction(patch)

    return round(float(eli), 4)  # result x.xxxx


def information_variation(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    # assert not np.array_equal(
    #     patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    iv = round(itt.information_variation(patch_1, patch_2), 4)  # result x.xxxx

    return iv


def binding_information(patch):
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    patch = patch.flatten()
    ib = itt.information_binding(patch)

    return round(float(ib), 4)  # result x.xxxx


def co_information(patch):
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a sigle dimensinoal array
    patch = patch.flatten()
    co_i = itt.information_co(patch)

    return round(float(co_i), 4)  # result x.xxxx



