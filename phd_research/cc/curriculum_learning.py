from map_measure import Measure, MeasureType,Ordering
from reconstructor import sort_patches_or_images
import tensorflow as tf 

class Curriculum(object):
    def __init__(self, training_batch, batch_size):
        self.syllabus = Syllabus()
        self.batch = training_batch
        self.measure_type = None
        self.batch_size = batch_size

    def Propose(self, measure=Measure.MI, ordering=Ordering.Ascending):
        assert tf.contrib.framework.is_tensor(self.batch)
        syllabus = sort_patches_or_images(self.batch, self.batch_size, measure, ordering)

        return syllabus

    def _Assess(self):
        pass