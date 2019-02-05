import tensorflow as tf
import numpy as np
tf_func = tf.py_func

class FM_POR(object):
    input = None
    input_size = None
    patch_size = None

    def __init__(input_feature_map, input_size, patch_size):
        assert input_feature_map not None, "Suplied feature map contains no data"
        assert isinstance(input_feature_map, np.ndarray)
        self.input = input_feature_map
        self.input_size = input_size
        self.patch_size = patch_size

    def Apply(): 
        assert input_size > patch_size, "Input size must be orders of magnitude larger than patch size"


