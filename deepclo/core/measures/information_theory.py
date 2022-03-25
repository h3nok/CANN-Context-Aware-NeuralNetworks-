import numpy as np
import tensorflow as tf
from deepclo.core.measures import itt


def entropy(patch):
    """
    In information theory, the entropy of a random variable is the average
    level of "information", "surprise", or "uncertainty" inherent to the
    variable's possible outcomes.

    When applied to discrete images, this measures how much relevant
    information is contained within an image when representing the image
    as a discrete information source

    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    Args:
        patch: a numpy array

    Returns: entropy of the array

    """
    patch_data = patch.flatten()
    e = itt.entropy(patch_data)

    return round(float(e), 4)  # result x.xxxx


def cross_entropy(patch_1, patch_2):
    """
    In information theory, the cross-entropy between two probability
    distributions p and q over the same underlying set of events measures
    the average number of bits needed to identify an event drawn from the
    set if a coding scheme used for the set is optimized for an estimated
    probability  distribution q, rather than the true distribution p.

    Args:
        patch_1: image patch_1, numpy array
        patch_2: image patch_2, numpy array

    Returns: cross entropy of the two images

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    if np.array_equal(patch_1, patch_2):
        print("Warning! Patches are binary equivalent, Cross entropy = 0.0")
        return 0.0

    # flatten the tensor
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()
    ce = itt.entropy_cross(patch_1, patch_2)
    return round(float(ce), 4)  # result x.xxxx


def cross_entropy_pmf(patch_1, patch_2):
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"

    if np.array_equal(patch_1, patch_2):
        print("Warning! Patches are binary equivalent, Cross entropy = 0.0")
        return 0.0

    # flatten the tensor into a single dimensional array
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()
    ce = itt.entropy_cross_pmf(patch_1, patch_2)

    return round(float(ce), 4)  # result x.xxxx


def conditional_entropy(patch_1, patch_2):
    """
    In information theory, the conditional entropy quantifies the amount
    of information needed to describe the outcome of a random variable
    Y given that the value of another random variable X is known.
    Here, information is measured in shannons, nats, or hartleys.

    https://en.wikipedia.org/wiki/Conditional_entropy
    Args:
        patch_1: numpy array
        patch_2: numpy array

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"

    if np.array_equal(patch_1, patch_2):
        print("Warning! Patches are binary equivalent, Cross entropy = 0.0")
        return 0.0

    # flatten the tensor
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()
    ce = itt.entropy_conditional(patch_1, patch_2)
    return round(float(ce), 4)  # result x.xxxx


def residual_entropy(patch_1):
    """

    https://en.wikipedia.org/wiki/Residual_entropy
    Args:
        patch_1:

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."

    patch_1 = patch_1.flatten()
    re = round(itt.entropy_residual(patch_1), 4)  # result x.xxxx

    return re


def joint_entropy(patch_1, patch_2):
    """
    In information theory, joint entropy is a measure of
    the uncertainty associated with a set of variables

    https://en.wikipedia.org/wiki/Joint_entropy

    Args:
        patch_1:
        patch_2:

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    if np.array_equal(patch_1, patch_2):
        print("Warning! Patches are binary equivalent, Joint entropy = 0.0")
        return 0.0
    # combine the two tensors into one
    patch_data = np.concatenate((patch_1, patch_2)).flatten()
    je = round(itt.entropy_joint(patch_data), 4)  # result x.xxxx

    return je


def kl_divergence(patch_1, patch_2):
    """
    In mathematical statistics, the Kullback–Leibler divergence,
    (also called relative entropy), is a statistical distance: a measure of how one probability
    distribution Q is different from a second, reference probability distribution P.
    A simple interpretation of the divergence of P from Q is the expected excess surprise from
    using Q as a model when the actual distribution is P.

    Args:
        patch_1:
        patch_2:

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"

    if np.array_equal(patch_1, patch_2):
        print("Warning! Patches are binary equivalent, KL Divergence = 0.0")
        return 0.0

    # combine the two tensors into one
    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    je = round(itt.divergence_kullbackleibler(patch_1, patch_2), 4)  # result x.xxxx

    return je


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
    """

    Args:
        patch_1:
        patch_2:

    Returns:

    """
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
    """

    Args:
        patch_1:

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."

    patch_1 = patch_1.flatten()

    ei = round(itt.information_enigmatic(patch_1), 4)  # result x.xxxx

    return ei


def lautum_information(patch_1, patch_2):
    """

    Args:
        patch_1:
        patch_2:

    Returns:

    """
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
    """

    Args:
        patch_1:

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."

    patch_1 = patch_1.flatten()

    ei = round(itt.information_multi(patch_1), 4)  # result x.xxxx

    return ei


def exogenous_local_information(patch):
    """

    Args:
        patch:

    Returns:

    """
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a single dimensional array
    patch = patch.flatten()
    eli = itt.information_exogenous_local(patch)

    return round(float(eli), 4)  # result x.xxxx


def information_interaction(patch):
    """

    Args:
        patch:

    Returns:

    """
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a single dimensional array
    patch = patch.flatten()
    eli = itt.information_interaction(patch)

    return round(float(eli), 4)  # result x.xxxx


def information_variation(patch_1, patch_2):
    """

    Args:
        patch_1:
        patch_2:

    Returns:

    """
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
    """

    Args:
        patch:

    Returns:

    """
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    patch = patch.flatten()
    ib = itt.information_binding(patch)

    return round(float(ib), 4)  # result x.xxxx


def co_information(patch):
    """

    Args:
        patch:

    Returns:

    """
    assert isinstance(patch, np.ndarray), "Patch data must be a numpy array."

    # flatten the tensor into a single dimensional array
    patch = patch.flatten()
    co_i = itt.information_co(patch)

    return round(float(co_i), 4)  # result x.xxxx

