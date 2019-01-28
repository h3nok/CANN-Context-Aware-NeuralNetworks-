import numpy as np
from pyitlib import discrete_random_variable as drv
import tensorflow as tf


def Entropy(patch):

    sess = tf.get_default_session()
    # flatten the tensor into a sigle dimensinoal array
    patch_data = sess.run(tf.reshape(patch, [-1]))

    e = round(drv.entropy(patch_data), 4)  # result x.xxxx

    return e
