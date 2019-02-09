import tensorflow as tf
import numpy as np
from patch_proposals import generate_patches_v2
from map_measure import Measure, MeasureType, Ordering
from reconstructor import reconstruct_from_patches
from PIL import Image
from cc_utils import ImageHelper as IMPLOT
tf_func = tf.py_func
tf_contrib = tf.contrib


class FM_POR(object):
    Input = None
    InputSize = None
    PatchSize = None
    Patches = None
    Ranker = None
    Output = None
    RankOrder = None

    def __init__(self, name, input_feature_map, input_size, patch_size, measure=Measure.MI,
                 order=Ordering.Ascending):
        # assert tf_contrib.framework.is_tensor(
        #     input_feature_map), "Input must be instance of tf.Tensor"
        assert isinstance(
            measure, Measure), "Supplied measure doesn't exist, measure: {}".format(measure.value)
        assert input_size > patch_size, "Input size must be orders of magnitude larger than patch size"
        assert input_size != 0, "Input shape must be > 0"
        assert patch_size != 0, "Patch shape must be > 0"

        self.Input = input_feature_map
        self.InputSize = input_size
        self.PatchSize = patch_size
        self.Ranker = measure 

    def Apply(self,input, dummy=''):
        self.Input = tf.convert_to_tensor(input)
        self.Patches = generate_patches_v2(self.Input, self.InputSize,
                                           self.InputSize, self.PatchSize, self.PatchSize)
        print (self.Patches)
        self.Output = reconstruct_from_patches(
            self.Patches, self.InputSize, self.InputSize, self.Ranker)
        return sess.run(self.Output)


def fm_por(name, input, input_size, patch_size, measure=Measure.MI, order=Ordering.Ascending):
    op = FM_POR(name, input, input_size, patch_size, measure, order)
    inputs = [input, '']
    return tf_func(op.Apply, inputs, tf.float32, name=name)


def test():
    slice56 = np.random.random((32, 32, 3))
    formatted = (slice56 * 255 / np.max(slice56)).astype('uint8')
    # fm_por_layer_1 = FM_POR('fm_por_1', formatted, 32, 8)
    # fm_por_layer_1.Apply()

    with tf.Session():
        output =  fm_por('fm_por_1', formatted, 32, 8)
        output = output.eval()
        plotter = IMPLOT()
        plotter.show_images([output,output],2,['output','output'])
    # img = Image.fromarray(formatted)
    # img.show()


if __name__ == '__main__':
    test()
