import statistics
from enum import Enum

import tensorflow as tf

import logger
from preprocessing.cc_preprocessing import decode_measure
from syllabus_factory.curriculum_factory import order_samples_or_patches
from syllabus_factory.map_measure import Measure, Ordering

_logger = logger.Configure(__file__, 'syllabus_factory.log')


class FitnessSignal(Enum):
    Stop = "Halts training when syllabus is unfit"
    Continue = "Syllabus is fit to continue training"
    Replace = "Replace syllabus by updating parameters"
    Unknown = "Fitness of a syllabus cannot be determined"


class SyllabusFactory(object):
    fitness_signal = None
    can_update = False
    backup_metrics = []
    ordering = None
    measure = None

    def __init__(self, training_batch, labels, batch_size, backup_metrics=None):
        assert labels is not None, "Must supply labels tensor with matching dimensions"
        _logger.debug("Constructing training curriculum")
        self.batch = training_batch
        self.measure_type = None
        self.batch_size = batch_size
        self.labels = labels
        self.fitness_signal = FitnessSignal.Continue
        if backup_metrics is not None:
            self.can_update = True
        else:
            self.can_update = False

    def update_syllabus(self):
        self.measure = self.backup_metrics[0]
        self.backup_metrics.remove(self.backup_metrics[0])

    def propose_syllabus(self, measure, ordering):
        _logger.debug("Proposing syllabus using content measure: {} and ordering: {}".format(measure, ordering))
        assert tf.contrib.framework.is_tensor(self.batch)
        self.ordering = ordering
        measure, ordering = decode_measure(measure, ordering)
        assert isinstance(measure, Measure)
        assert isinstance(ordering, Ordering)
        input_path, labels = order_samples_or_patches(
            self.batch, self.batch_size, measure, ordering,
            curriculum=True, labels=self.labels)

        return input_path, labels

    def evaluate_syllabus(self, losses, pi, baseline_threshold):
        _logger.debug("Evaluating syllabus, pi={}, beta={}...".format(pi, baseline_threshold))
        assert isinstance(losses, list), "Must supply a list of losses\n"

        if len(losses) != pi:
            _logger.warn("Evaluating syllabus using number of losses less than pi")

        syllabus_loss = statistics.mean(losses)

        if syllabus_loss > baseline_threshold:
            self.fitness_signal = FitnessSignal.stop
        elif syllabus_loss > baseline_threshold and self.can_update:
            self.fitness_signal = FitnessSignal.Replace
            self.update_syllabus()
        else:
            self.fitness_signal = FitnessSignal.Continue


