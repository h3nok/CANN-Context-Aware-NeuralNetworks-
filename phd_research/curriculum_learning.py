from cc.map_measure import Measure, MeasureType, Ordering, MEASURE_MAP
from cc.reconstructor import sort_patches_or_images
from preprocessing.cc_preprocessing import decode_measure
import tensorflow as tf

class Curriculum(object):
    def __init__(self, training_batch, labels, batch_size):
        self.batch = training_batch
        self.measure_type = None
        self.batch_size = batch_size
        self.labels = labels

    def Propose(self, measure, ordering):
        assert tf.contrib.framework.is_tensor(self.batch)
        measure, ordering = decode_measure(measure, ordering)
        assert isinstance(measure, Measure)
        assert isinstance(ordering, Ordering)
        syllabus = sort_patches_or_images(
            self.batch, self.batch_size, measure, ordering)

        return syllabus

    def _Assess(self):
        pass
