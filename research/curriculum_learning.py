from cc.map_measure import Measure, Ordering
from cc.curriculum_factory import order_samples_or_patches
from preprocessing.cc_preprocessing import decode_measure
import tensorflow as tf
import logger

_logger = logger.Configure(__file__, 'cc.log')


class SyllabusFactory(object):
    def __init__(self, training_batch, labels, batch_size):
        assert labels is not None, "Must supply labels tensor with matching dimensions"
        _logger.debug("Constructing training curriculum")
        self.batch = training_batch
        self.measure_type = None
        self.batch_size = batch_size
        self.labels = labels

    def propose_syllabus(self, measure, ordering):
        _logger.debug("Proposing syllabus using content measure: {} and ordering: {}".format(measure, ordering))
        assert tf.contrib.framework.is_tensor(self.batch)
        measure, ordering = decode_measure(measure, ordering)
        assert isinstance(measure, Measure)
        assert isinstance(ordering, Ordering)
        input_path, labels = order_samples_or_patches(
            self.batch, self.batch_size, measure, ordering, curriculum=True, labels=self.labels)

        return input_path, labels

