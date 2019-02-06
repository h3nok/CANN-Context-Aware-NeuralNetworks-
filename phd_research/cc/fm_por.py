import tensorflow as tf
import numpy as np
from patch_proposals import generate_patches_v2
from map_measure import Measure, MeasureType
from reconstructor import reconstruct_from_patches
from PIL import Image
tf_func = tf.py_func
tf_contrib = tf.contrib


class FM_POR(object):
    input = None
    input_size = None
    patch_size = None
    patches = None
    measure = None

    def __init__(self, name, input_feature_map, input_size, patch_size, measure=Measure.MI):
        assert tf_contrib.framework.is_tensor(input_feature_map), "Input must be instance of tf.Tensor"
        assert isinstance(
            measure, Measure), "Supplied measure doesn't exist, measure: {}".format(measure.value)
        assert input_size > patch_size, "Input size must be orders of magnitude larger than patch size"
        assert input_size != 0, "Input shape must be > 0"
        assert patch_size != 0, "Patch shape must be > 0"

        self.input = input_feature_map
        self.input_size = input_size
        self.patch_size = patch_size

    def Apply(self):
        self.patches = generate_patches_v2(self.input, self.input_size,
                                           self.input_size, self.patch_size, self.patch_size)

        print (self.patches)

def test():
    slice56 = np.random.random((32, 32,3))
    formatted = tf.convert_to_tensor((slice56 * 255 / np.max(slice56)).astype('uint8'))
    fm_por_layer_1 = FM_POR('fm_por_1', formatted, 32, 8)
    fm_por_layer_1.Apply()
    # img = Image.fromarray(formatted)
    # img.show()


if __name__ == '__main__':
    test()
