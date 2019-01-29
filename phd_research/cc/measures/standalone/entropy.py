import numpy as np
from pyitlib import discrete_random_variable as drv
import tensorflow as tf


def Entropy(patch):

    # flatten the tensor into a sigle dimensinoal array
    patch_data = patch.flatten()
    entropy = drv.entropy(patch_data)
    return round(float(entropy), 4)  # result x.xxxx
