from numpy.linalg import norm
from numpy import inf
import tensorflow as tf


def MaxNorm(patch_1, patch_2):
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert patch_1 != patch_2, "Patches are binary equivalent, Distance = 0"

    sess = tf.get_default_session()

    # combine the two tensors into one
    patch_data = sess.run(
        tf.concat([patch_1, patch_2], 0))

    return round(norm(patch_data, inf))
