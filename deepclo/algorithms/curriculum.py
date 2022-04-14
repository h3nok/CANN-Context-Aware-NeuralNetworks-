import numpy as np

import tensorflow as tf

from deepclo.core.measures.measure_functions import Measure, MeasureType, RANK_MEASURES
from deepclo.algorithms.por import assess_and_rank_images, sort_images, determine_measure_classification
from deepclo.utils import show_images, hist


class Curriculum:

    def __init__(self, batch: np.ndarray = None, labels: np.ndarray = None):
        """
        We describe a quantitative and practical framework to integrate Curriculum Learning (CL)
        into deep learning training pipeline to improve feature learning in deep feed-forward
        networks. The framework has several unique characteristics: 1. dynamicity – it proposes
        a set of batch-level training strategies (syllabi or curricula) that are sensitive to
        data complexity 2. adaptivity – it dynamically estimates the effectiveness of a given
        strategy and performs objective comparison with alternative strategies making the method
        suitable both for practical and research purposes. 3. employs replace-retrain mechanism
        when a strategy is unfit to the task at hand. In addition to these traits, the framework
        can combine CL with several variants of gradient descent (GD) algorithms and has been
        used to generate efficient batch specific or data-set specific strategies. Comparative
        studies of various current state-of-the-art vision models such as FixEfficientNet and BiT-L
        (ResNet) on several benchmark datasets including CIFAR10 demonstrate the effectiveness of
        the proposed method. We present results that show training loss reduction by as much as a
        factor 5. Additionally, we present a set of practical curriculum strategies to improve the
        generalization performance of select networks on various datasets.

        Args:
            batch: a batch of training data of shape (n, w, h, c)

        """
        self.batch = batch
        self.labels = labels

        # assert len(self.batch.shape) == 4
        # assert len(self.labels.shape) == 2
        # assert self.batch.shape[0] == self.labels.shape[0]

        self._measure = None
        self._rank_ordering = 0
        self._reference_image_index = None
        self.ranks = np.array([])
        self.input_order = np.array([])
        self.sorted_batch = np.array([])
        self.sorted_labels = np.array([])

        self.measure_type = None

    @property
    def measure(self):
        return self._measure

    @measure.setter
    def measure(self, value):
        self._measure = RANK_MEASURES[value.upper()]

    @property
    def rank_order(self):
        return self._rank_ordering

    @rank_order.setter
    def rank_order(self, value):
        assert value in [0, 1, 'asc, des']
        self._rank_ordering = value

    @property
    def reference_image_index(self):
        return self._reference_image_index

    @reference_image_index.setter
    def reference_image_index(self, value):
        self._reference_image_index = value

    def _select_low_entropy_reference_image(self):
        """

        Returns: np.ndarray

        """
        entropy_ranks = assess_and_rank_images(self.batch,
                                               content_measure=Measure.ENTROPY,
                                               reference_block_index=None)
        return np.argmin(entropy_ranks)

    def _rank_and_sort_batch(self,
                             measure: Measure = None,
                             reference_imag_index: int = None,
                             image_ordering: int = 0):
        """
        Rank images using a standalone or a similarity or distance measure

        Args:
            measure:
            reference_imag_index:

        Returns:

        """
        if not measure:
            assert self._measure
        else:
            self._measure = measure

        self.measure_type = determine_measure_classification(measure)

        if self.measure_type == MeasureType.DISTANCE and not reference_imag_index:
            print("Reference image index not supplied. Using minimum entropy sample as a reference image ")
            self._reference_image_index = self._select_low_entropy_reference_image()
        else:
            self._reference_image_index = reference_imag_index

        self.ranks = assess_and_rank_images(self.batch,
                                            content_measure=measure,
                                            reference_block_index=self._reference_image_index)

        self.sorted_batch, self.sorted_labels, self.input_order = sort_images(batch_or_image_blocks=self.batch,
                                                                              ranks=self.ranks,
                                                                              labels=self.labels,
                                                                              block_rank_ordering=image_ordering)
        return self.sorted_batch, self.sorted_labels

    def syllabus(self,
                 measure: Measure = None,
                 reference_imag_index: int = None,
                 image_ordering: int = 0):
        """
        Generate batch training syllabus

        Args:
            image_ordering: Ordering of samples selected from (0: ascending, 1: descending)
            reference_imag_index: reference image index if measure is a similarity metric
            measure: Measure function used to generate a training syllabus

        Returns: np.array of sorted blocks

        """

        return self._rank_and_sort_batch(measure=measure,
                                         reference_imag_index=reference_imag_index,
                                         image_ordering=image_ordering)

    def _propose_syllabus(self, batch: np.ndarray, labels: np.ndarray):

        self.batch = batch
        self.labels = labels

        assert self._measure
        assert self._rank_ordering in [0, 1, 'dec', 'asc']

        return self._rank_and_sort_batch(self._measure, self._reference_image_index, self._rank_ordering)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, 3],
                                                dtype=tf.uint8),
                                  tf.TensorSpec(shape=[None, 1], dtype=tf.uint8)))
    def generate_syllabus(self, batch: np.ndarray, labels: np.ndarray):
        """
        Generate batch training syllabus - tensorflow training pipeline

        Args:
            batch: np.ndarray -  a batch of images
            labels: np.ndarray - a set of labels

        Returns:

        """
        batch_syllabus, batch_labels = tf.numpy_function(self._propose_syllabus, (batch, labels), [tf.uint8, tf.uint8], name='CLO')

        batch_syllabus.set_shape(batch.get_shape())
        batch_labels.set_shape(labels.get_shape())

        return batch_syllabus, batch_labels

    def print_syllabus(self):
        """

        Returns:

        """
        titles = []

        if self.ranks.any():
            titles = [f"{self._measure}:{self.ranks[index]}, "
                      f"rank: {np.where(self.input_order == index)[0]}"
                      for index in range(self.batch.shape[0])]

            show_images(self.batch, titles=titles)
        else:
            show_images(self.batch, titles=titles)

    def plot_rank_distribution(self, dataset_name):
        """
        Plot rank distribution of the b
        Args:
            dataset_name:

        Returns:

        """
        measure = str(self._measure).split('.')[1]
        title = f"{measure} Distribution of {dataset_name}"

        if self.measure_type == MeasureType.DISTANCE:
            self.ranks = self.ranks[self.ranks != 0.]

        hist(self.ranks, title=title, x_label=measure)
