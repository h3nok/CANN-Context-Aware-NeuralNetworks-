import statistics
from enum import Enum
import tensorflow as tf
import numpy as np
import logger
from preprocessing.cc_preprocessing import decode_measure
from syllabus_factory.curriculum_factory import order_samples_or_patches
from syllabus_factory.map_measure import Measure, Ordering, MeasureType, map_measure_fn
from tqdm import tqdm

_logger = logger.configure(__file__, 'syllabus_factory.log')


class FitnessSignal(Enum):
    Stop = "Halts training when syllabus is unfit"
    Continue = "Syllabus is fit to continue training"
    Replace = "Replace syllabus by updating parameters"
    Unknown = "Fitness of a syllabus cannot be determined"


def _determine_measure_type(measure):
    assert isinstance(measure, Measure), "{} not found.".format(measure.value)
    if measure in [Measure.JE, Measure.MI, Measure.CE, Measure.L1, Measure.L2,
                   Measure.MAX_NORM, Measure.KL, Measure.SSIM, Measure.PSNR,
                   Measure.MI_NORMALIZED, Measure.LI, Measure.IV, Measure.CROSS_ENTROPY,
                   Measure.CROSS_ENTROPY_PMF]:

        return MeasureType.Dist
    else:
        return MeasureType.STA


def sort_by_content_measure(patches_data, measure_fn, labels, curriculum=False):
    """[summary]

    Arguments:
        patches_data {np.ndarray} -- the image patches in tensor format
        measure_fn {Measure} -- STA measure function to apply for sorting

    Keyword Arguments:
        ordering {Ordering} -- [description] (default: {Ordering.Ascending})
    """

    _logger.info("Entering sort patches by content measure ...")

    assert isinstance(
        patches_data, np.ndarray), "Supplied data must be instance of np.ndarray"

    number_of_patches = patches_data.shape[0]

    sorted_patches = None
    sorted_labels = None

    if not curriculum:
        sorted_patches = np.array(
            sorted(patches_data, key=lambda patch: measure_fn(patch)))
    else:
        patches_data = zip(patches_data, labels)
        sorted_patches, sorted_labels = np.array(zip(*sorted(patches_data,
                                                             key=lambda patch: measure_fn(patches_data[patch]))))

    assert len(
        sorted_patches) == number_of_patches, \
        _logger.error("Loss of data when sorting patches data")
    assert patches_data.shape == sorted_patches.shape, \
        _logger.error("Original tensor and sorted tensor have different shapes")

    _logger.info("Successfully sorted patches using content measure ")

    sorted_labels = tf.convert_to_tensor(sorted_labels, dtype=tf.int8)
    sorted_patches = tf.convert_to_tensor(sorted_patches, dtype=tf.float32)

    if curriculum:
        return sorted_patches, sorted_labels

    return sorted_patches


class SyllabusFactory(tf.Module):
    fitness_signal = None
    can_update = False
    backup_metrics = []
    ordering = None
    measure = None
    graph = None

    def __init__(self, graph, training_batch, labels, batch_size, backup_metrics=None):
        super().__init__()
        assert labels is not None, "Must supply names tensor with matching dimensions"
        _logger.debug("Constructing training curriculum")
        tf.logging.debug("Constructing training curriculum")
        self.batch = training_batch
        self.measure_type = None
        self.batch_size = batch_size
        self.labels = labels
        self.fitness_signal = FitnessSignal.Continue
        self.graph = graph
        if backup_metrics is not None:
            self.can_update = True
        else:
            self.can_update = False

        assert graph

    @tf.function
    def update_syllabus(self):
        self.measure = self.backup_metrics[0]
        self.backup_metrics.remove(self.backup_metrics[0])

    @tf.function
    def propose_syllabus(self, measure, ordering):
        _logger.debug("Proposing syllabus using content measure: {} and ordering: {}".format(measure, ordering))
        tf.logging.debug("Proposing syllabus using content measure: {} and ordering: {}".format(measure, ordering))
        assert tf.contrib.framework.is_tensor(self.batch)
        self.ordering = ordering
        measure, ordering = decode_measure(measure, ordering)
        assert isinstance(measure, Measure)
        assert isinstance(ordering, Ordering)
        input_path, labels = self.generate_syllabus(measure, ordering, curriculum=True)

        return input_path, labels

    @tf.function
    def evaluate_syllabus(self, losses, pi, baseline_threshold):
        _logger.debug("Evaluating syllabus, pi={}, beta={}...".format(pi, baseline_threshold))
        tf.logging.debug("Evaluating syllabus, pi={}, beta={}...".format(pi, baseline_threshold))
        assert isinstance(losses, list), "Must supply a list of losses\n"

        if len(losses) != pi:
            _logger.warn("Evaluating syllabus using number of losses less than pi")
            tf.logging.warn("Evaluating syllabus using number of losses less than pi")

        syllabus_loss = statistics.mean(losses)

        if syllabus_loss > baseline_threshold:
            self.fitness_signal = FitnessSignal.stop
        elif syllabus_loss > baseline_threshold and self.can_update:
            self.fitness_signal = FitnessSignal.Replace
            self.update_syllabus()
        else:
            self.fitness_signal = FitnessSignal.Continue

    @tf.function
    def generate_syllabus(self, measure=None, ordering=None, curriculum=True):
        """[summary]

        Arguments:
            patches_data {tensor} -- tensor having shape [number_of_patches, height,width,channel]
            total_patches {int} -- total number of patches

        Keyword Arguments:
            measure {Measure} -- ranking measure to use for sorting (default: {Measure.JE})
            ordering {Ordering} -- sort order (default: {Ordering.Ascending})
        """
        assert measure
        coord = tf.train.Coordinator()
        # TODO - parallel implementation
        _logger.info("Entering Generate syllabus ... ")
        measure_type = _determine_measure_type(measure)

        measure_fn = map_measure_fn(measure, measure_type)
        with tf.Session(graph=self.graph) as sess:
            tf.train.start_queue_runners(sess=sess, coord=coord)
            self.batch = sess.run(self.batch)
            labels_data = None
            if curriculum:
                if self.labels is not None:
                    labels_data = sess.run(self.labels)
            sess.close()

        if measure_type == MeasureType.STA:
            _logger.info(
                'Measure type is standalone, calling _sort_patches_by_content_measure ...')
            return sort_by_content_measure(self.batch, measure_fn, self.labels)

        if measure_type != MeasureType.Dist:
            _logger.error(
                "Supplied measure is not distance measure, please call _sort_patches_by_standalone_measure instead")

        _logger.info(
            "Sorting patches by distance measure, measure: {}".format(measure.value))

        def _compare_numpy(reference_patch, patch):
            patches_to_compare = (reference_patch, patch)
            dist = measure_fn(patches_to_compare)
            return dist

        def _swap(i, j):
            # print("Swapping %d with %d" % (i, j))
            self.batch[[i, j]] = self.batch[[j, i]]

        def _swap_labels(i, j):
            labels_data[[i, j]] = labels_data[[j, i]]

        for i in tqdm(range(0, self.batch_size)):
            _logger.info("Sorting batch , measure: {}, # of samples: {}".format(measure.value, self.batch_size))
            # TODO- make configurable
            closest_distance_thus_far = 100
            reference_patch_data = self.batch[i]  # set reference patch
            # sorted_patches.append(reference_patch_data)

            # compare the rest to reference patch
            for j in range(i + 1, self.batch_size):
                # print ("Comparing %d and %d" %(i,j))
                distance = _compare_numpy(
                    reference_patch_data, self.batch[j])
                if j == 1:
                    closest_distance_thus_far = distance
                    continue
                if ordering == Ordering.Ascending and distance < closest_distance_thus_far:
                    closest_distance_thus_far = distance
                    _swap(i + 1, j)
                    if curriculum:
                        _swap_labels(i + 1, j)
                    # reference_patch_data = patches_data[i]
                elif ordering == Ordering.Descending and distance > closest_distance_thus_far:
                    closest_distance_thus_far = distance
                    _swap(i + 1, j)
                    if self.labels is not None:
                        _swap_labels(i + 1, j)

        sorted_patches = tf.convert_to_tensor(self.batch, dtype=tf.float32)
        sorted_labels = None
        if curriculum:
            sorted_labels = tf.convert_to_tensor(labels_data, tf.uint8)
            assert sorted_labels.shape[0] == self.batch_size

        assert sorted_patches.shape[0] == self.batch_size, _logger.error("Sorted patches list contains more or less \
            number of patches compared to original")
        _logger.info(
            "Successfully sorted patches, closing session and exiting ...")

        return sorted_patches, sorted_labels
