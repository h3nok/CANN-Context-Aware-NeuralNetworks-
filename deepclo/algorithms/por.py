import os

from PIL import Image
from skimage.util.shape import view_as_blocks

from deepclo.algorithms.image_processsing import sort_images, construct_new_input, assess_and_rank_images
from deepclo.core.measures.information_theory import *
from deepclo.core.measures.measure_functions import Measure, RANK_MEASURES, determine_measure_classification, \
    MeasureType
from deepclo.utils import show_images, configure_logger


class POR:

    def __init__(self,
                 sample: np.ndarray = np.array([]),
                 label: np.ndarray = np.array([]),
                 block_shape: tuple = None
                 ):
        """
        Patch Ordering and Reconstruction (POR) Algorithm. The function
        Generate-Patches generates equal sized patches of the input image. Compute-
        Individual-Index calculates the index of a given patch when the MeasureType
        is of type standalone while Compute-Mutual-Index computes an index of
        similarity between two patches. Sort-Patches sorts the patches according to
        indices and Reconstruct-Sample constructs a sample using sorted patches. For
        computational efficiency patch_size is taken from (4x4, 8x8, 16x16) and all
        samples are resized to 32x32 prior to preprocessing. Since the dataset consists
        of color (RGB) images the algorithm computes the index of each channel and
        returns the average .

        Args:
            label (object): np.ndarray
            sample: np.ndarray
            block_shape: tuple (w, h)
        """
        self.reconstructed_input = None
        self.label = label
        self.sample = sample
        self.width = None
        self.height = None
        self.channels = None
        if self.sample.any():
            assert len(self.sample.shape) == 3
            self.width = self.sample.shape[0]
            self.height = self.sample.shape[1]
            self.channels = self.sample.shape[2]

        if self.label.any():
            assert len(self.label.shape) == 1

        self._block_shape = block_shape
        self.blocks = np.array([])
        self.sorted_blocks = None
        self.ranks = None
        self.ranks = np.array([])
        self.patches_per_row = None
        self.patches_per_col = None
        self._block_raking_measure = None
        self._rank_ordering = 1
        self._logger = configure_logger(self.__class__.__name__, console=False)

    def _divide_into_blocks(self, block_shape):
        """
        split the image into equal sized blocks

        Args:
            block_shape: (M x N )
        Returns:

        """

        self.patches_per_row = self.height // block_shape[0]
        self.patches_per_col = self.width // block_shape[1]

        self._logger.debug(f"Splitting a {self.width}x{self.height} image into :"
                           f"{self.patches_per_row * self.patches_per_col}, {block_shape[0]} :"
                           f"by {block_shape[1]} blocks")

        block_views = view_as_blocks(self.sample, block_shape)
        blocks = []

        # TODO - optimize with native numpy operation
        for i, j in enumerate(block_views):
            for block in j:
                blocks.append(block[0])

        self.blocks = np.asarray(blocks)

        assert self.blocks[0].shape == (block_shape[0], block_shape[1], self.channels)

        self._validate()

    def _validate(self):
        """
        Ensure split_into_blocks preserves pixel composition

        Returns: throws exception if validation fails

        """
        sample_set = set(self.sample.flatten())
        for block in self.blocks:
            if not set(block.flatten()).issubset(sample_set):
                raise RuntimeError("Unable to validate  POR operation. block is not a subset of original image")

    @property
    def measure(self):
        return self._block_raking_measure

    @measure.setter
    def measure(self, measure):
        measure = measure.upper()
        if measure not in list(RANK_MEASURES.keys()):
            self._logger.debug(list(RANK_MEASURES.keys()))
            raise RuntimeError(f"Invalid measure name '{measure}'. Please select one from the list")

        self._block_raking_measure = RANK_MEASURES[measure]

    @property
    def block_shape(self):
        return self._block_shape

    @block_shape.setter
    def block_shape(self, block_shape: tuple):
        self._block_shape = block_shape

    @property
    def rank_order(self):
        return self._rank_ordering

    @rank_order.setter
    def rank_order(self, value):
        assert value in [0, 1, 'asc', 'des']
        self._rank_ordering = value

    def _select_maximum_low_entropy_reference_block(self):
        """

        Returns: np.ndarray

        """
        entropy_ranks = assess_and_rank_images(self.blocks,
                                               content_measure=Measure.ENTROPY,
                                               reference_block_index=None)
        return np.argmin(entropy_ranks)

    def split_image(self, block_shape: tuple = None):
        """
        Split image into equal sized blocks

        Args:
            block_shape:

        Returns:
        """
        if not block_shape:
            assert self._block_shape, "Must supply block_shape for algorithm POR"

        elif block_shape:
            self._block_shape = block_shape

        self._block_shape = (self._block_shape[0],
                             self._block_shape[1],
                             self.channels)

        self._divide_into_blocks(block_shape=self._block_shape)

        assert len(self.blocks) > 0

    def construct_new_input_from_blocks(self,
                                        block_ranking_measure: Measure = None,
                                        rank_order=None,
                                        reference_block_index=0):
        """
        Build a new input by combining image blocks according to rank values and ordering of those ranks.

        Args:
            rank_order: ordering of blocks ( 0: asc, 1: dec)
            block_ranking_measure: measure to use to calculate ranks of blocks
            reference_block_index: reference image to be used to rank distance between blocks when measure is of type
            DISTANCE

        Returns: reconstructed input and its label both numpy arrays

        """

        assert self.blocks.any(), "Invalid input. Blocks array is empty. Must call split_image before " \
                                  "construct_new_input "
        if not block_ranking_measure:
            assert self._block_raking_measure, "Please set block rank measure,  por.measure = ..."
        else:
            self._block_raking_measure = block_ranking_measure

        if not rank_order:
            assert self._rank_ordering in [0, 1]
        else:
            self._rank_ordering = rank_order

        metric_type = determine_measure_classification(self._block_raking_measure)

        if metric_type == MeasureType.STANDALONE:
            self._logger.debug(f"Constructing new input using {self._block_raking_measure} STA measure... ")
            self.ranks = assess_and_rank_images(self.blocks, self._block_raking_measure, reference_block_index=None)
            self.sorted_blocks, _ = sort_images(self.blocks,
                                                ranks=self.ranks,
                                                block_rank_ordering=self._rank_ordering)

            self.reconstructed_input = construct_new_input(self.sorted_blocks,
                                                           self.height,
                                                           self.width,
                                                           self.channels)

        elif metric_type == MeasureType.DISTANCE:
            if not reference_block_index:
                self._logger.warning("Reference image index not supplied. Using minimum entropy sample as a reference "
                                     "image ")
                reference_block_index = self._select_maximum_low_entropy_reference_block()

            self._logger.debug(f"Constructing new input using {self._block_raking_measure} DISTANCE measure, "
                               f"reference_block_index: {reference_block_index}... ")
            self.ranks = assess_and_rank_images(self.blocks,
                                                content_measure=self._block_raking_measure,
                                                reference_block_index=reference_block_index)
            self.sorted_blocks, _ = sort_images(self.blocks,
                                                ranks=self.ranks,
                                                block_rank_ordering=self._rank_ordering)

            assert not np.array_equal(self.sorted_blocks, self.blocks)

            self.reconstructed_input = construct_new_input(self.sorted_blocks,
                                                           self.height,
                                                           self.width,
                                                           self.channels)

        return self.reconstructed_input, self.label

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 3],
                                                dtype=tf.uint8),
                                  tf.TensorSpec(shape=[1, ], dtype=tf.uint8)))
    def algorithm_por(self, sample, label):
        """
        Tensorflow training pipeline operator

        Args:
            sample: tf.tensor representing a sample
            label: tf.tensor label

        Returns: preprocessed sample and its label

        """
        new_sample, new_label = tf.numpy_function(self.preprocess_input,
                                                  (sample, label),
                                                  [tf.uint8, tf.uint8],
                                                  name='A-POR')
        new_sample.set_shape(sample.get_shape())
        new_label.set_shape(label.get_shape())

        return new_sample, label

    def preprocess_input(self, sample: np.ndarray, label: np.ndarray):
        """
        Preprocessing API - to be used in training pipeline

        Args:
            sample:
            label:

        Returns:

        """
        self._logger.debug("Entering preprocess_input, proper function .... ")

        self.sample = sample
        self.label = label

        if len(self.sample.shape) == 4:
            self.width = self.sample.shape[1]
            self.height = self.sample.shape[2]
            self.channels = self.sample.shape[3]
        else:
            self.width = self.sample.shape[0]
            self.height = self.sample.shape[1]
            self.channels = self.sample.shape[2]

        self.split_image(self.block_shape)

        return self.construct_new_input_from_blocks()

    def save_image_blocks(self, output_dir: str = None) -> None:
        """
        Write image blocks as png to folder

        Args:
            output_dir:

        Returns:

        """
        if output_dir:
            output_dir = os.path.join(output_dir, str(self.label[0]))
        else:
            output_dir = str(self.label[0])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        counter = 0
        for block in list(self.blocks):
            img = Image.fromarray(block, 'RGB')
            img.save(os.path.join(output_dir, f'{counter}.png'))
            counter += 1

        Image.fromarray(self.sample,
                        'RGB').save(os.path.join(output_dir,
                                                 str(self.label[0]) + ".png"))

    def show_image_blocks(self):
        show_images(images=self.blocks, titles="Image Blocks")

    def save_reconstructed_input(self,
                                 measure,
                                 output_dir: str = None):
        """
        Write reconstructed input to disc

        Args:
            measure:
            output_dir:

        Returns:

        """
        if output_dir:
            output_dir = os.path.join(output_dir, str(self.label[0]))
        else:
            output_dir = str(self.label[0])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if not measure:
            measure = self._block_raking_measure

        patch_size_str = f"{self._block_shape[0]}x{self._block_shape[1]}"
        filename = os.path.join(output_dir, str(self.label[0]) + f"_{measure}_{patch_size_str}_reconstructed.png")
        Image.fromarray(self.reconstructed_input,'RGB').save(filename)
        self._logger.debug(f"Successfully saved reconstructed input, filepath: {filename}")

        return filename

    def show_original(self):
        from matplotlib import pyplot as plt
        plt.grid(False)

        # Hide axes ticks
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.sample, interpolation='nearest', aspect='auto')
        plt.show()

    def show_reconstructed_input(self):
        from matplotlib import pyplot as plt
        plt.grid(False)
        # Hide axes ticks
        plt.xticks([])
        plt.yticks([])
        plt.imshow(self.reconstructed_input, interpolation='nearest', aspect='auto')
        plt.show()
