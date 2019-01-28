import numpy as np
from pyitlib import discrete_random_variable as drv
import tensorflow as tf


def MutualInformation(patch_1, patch_2):

    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert patch_1 != patch_2, "Patches are binary equivalent, Distance = 0"

    sess = tf.get_default_session()

    # combine the two tensors into one
    # flatten the tensor into a sigle dimensinoal array
    patch_1 = sess.run(tf.reshape(patch_1, [-1]))
    patch_2 = sess.run(tf.reshape(patch_2, [-1]))

    mi = round(drv.information_mutual(patch_1, patch_2), 4)  # result x.xxxx

    return mi
