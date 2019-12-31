import matplotlib.pyplot as plt
import tensorflow as tf

from map_measure import Measure, Ordering
from patch_proposals import p_por
from utils import ImageHelper as IMPLOT

tf_func = tf.py_func
tf_contrib = tf.contrib


class FmPor(object):
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
        self.RankOrder = order

    def apply(self, input, dummy=''):
        self.Input = tf.convert_to_tensor(input)
        output = p_por(self.Input, self.InputSize, self.InputSize,
                       self.Ranker, self.RankOrder, self.PatchSize, self.PatchSize)

        return output


def fm_por(name, input, input_size, patch_size, measure=Measure.MI, order=Ordering.Ascending):
    op = FmPor(name, input, input_size, patch_size, measure, order)
    inputs = [input, '']
    # return op.apply(input)
    # output = np.array([input_size,input_size, 3],np.float32)
    return tf_func(op.apply, inputs, tf.float32, name=name)
    # return output


def run():
    image_file = 'samples/husky.jpg'
    patch_width = 56
    patch_height = 56
    input_size = (224, 224)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image = tf.gfile.FastGFile(image_file, 'rb').read()
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        output = fm_por('fm_por_1', image.eval(), input_size[0], patch_width)
        output = sess.run(output)
        plotter = IMPLOT()
        show_images([image.eval(), output], 2, ['input', 'output'])
        plt.show()


if __name__ == '__main__':
    run()
