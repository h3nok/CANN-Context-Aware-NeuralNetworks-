import os

from PIL import Image
from skimage.util.shape import view_as_blocks

from deepclo.algorithms.image_processsing import measure_image_content, \
    sort_image_blocks, measure_content_similarity, construct_new_input
from deepclo.core.measures.information_theory import *
from deepclo.core.measures.measure_functions import Measure, \
    determine_measure_classification, MeasureType


def rank_image_blocks(blocks, content_measure: Measure, reference_block_index: int = None):
    """
    Rank image blocks by measuring the content of each block or similarity of each block to
    the reference block

    Args:
        blocks:
        content_measure:
        reference_block_index:

    Returns:

    """
    ranks = []
    if determine_measure_classification(content_measure) == MeasureType.STANDALONE:
        for i, block in enumerate(blocks):
            rank = measure_image_content(block, content_measure=content_measure)
            ranks.append(rank)

        return np.array(ranks)

    else:
        assert reference_block_index >= 0
        ranks = []
        reference_block = blocks[reference_block_index]

        for i, block in enumerate(blocks):
            if i == reference_block_index:
                continue
            rank = measure_content_similarity(reference_block, block, content_measure)
            ranks.insert(i, rank)

        ranks.insert(reference_block_index, 0.0)

        return np.array(ranks)


class AlgorithmPOR:

    def __init__(self, sample: np.ndarray, label: np.ndarray, block_shape):
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
        assert len(self.sample.shape) == 3
        assert len(self.label.shape) == 1
        self.width = self.sample.shape[0]
        self.height = self.sample.shape[1]
        self.channels = self.sample.shape[2]
        self.block_shape = block_shape
        self.blocks = np.array([])
        self.sorted_blocks = None
        self.block_ranks = None
        self.ranks = np.array([])
        self.patches_per_row = None
        self.patches_per_col = None
        self.measure = None

    def split_image(self, block_shape: tuple = None):
        """
        Split image into equal sized blocks

        Args:
            block_shape:
        Returns:
        """
        if not block_shape:
            assert self.block_shape

        elif block_shape:
            self.block_shape = block_shape

        assert self.block_shape
        assert len(self.block_shape) == 2
        assert self.block_shape[0] == self.block_shape[1]
        self.block_shape = (self.block_shape[0], self.block_shape[1], self.channels)

        self._divide_into_blocks(block_shape=self.block_shape)

        assert len(self.blocks) > 0

    def _divide_into_blocks(self, block_shape):
        """
        split the image into equal sized blocks

        Args:
            block_shape: (M x N )
        Returns:

        """
        self.patches_per_row = self.height // block_shape[0]
        self.patches_per_col = self.width // block_shape[1]

        print(f"Splitting a {self.width}x{self.height} image into {self.patches_per_row * self.patches_per_col}, "
              f"{block_shape[0]} by {block_shape[1]}")

        block_views = view_as_blocks(self.sample, self.block_shape)
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
            assert set(block.flatten()).issubset(sample_set)

    def construct_new_input(self,
                            block_ranking_measure: Measure = Measure.ENTROPY,
                            rank_order=0):
        """
        Build a new input by combining image blocks according to rank values and ordering of those ranks.

        Args:
            block_ranking_measure:
            rank_order:

        Returns:

        """

        assert self.blocks.any(), "Invalid input. Blocks array is empty. Must call split_image before " \
                                  "construct_new_input "
        self.measure = block_ranking_measure
        metric_type = determine_measure_classification(block_ranking_measure)

        if metric_type == MeasureType.STANDALONE:
            self.block_ranks = rank_image_blocks(self.blocks, block_ranking_measure)
            self.sorted_blocks, _ = sort_image_blocks(self.blocks,
                                                      ranks=self.block_ranks,
                                                      block_rank_ordering=rank_order)

            assert not np.array_equal(self.sorted_blocks, self.blocks)

            self.reconstructed_input = construct_new_input(self.sorted_blocks,
                                                           self.height,
                                                           self.width,
                                                           self.channels)

        elif metric_type == MeasureType.DISTANCE:
            reference_block_index = 2
            self.block_ranks = rank_image_blocks(self.blocks,
                                                 content_measure=block_ranking_measure,
                                                 reference_block_index=reference_block_index)
            self.sorted_blocks, _ = sort_image_blocks(self.blocks,
                                                      ranks=self.block_ranks,
                                                      block_rank_ordering=rank_order)

            assert not np.array_equal(self.sorted_blocks, self.blocks)
            self.reconstructed_input = construct_new_input(self.sorted_blocks,
                                                           self.height,
                                                           self.width,
                                                           self.channels)

        return self.reconstructed_input

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

    def save_reconstructed_input(self, measure, output_dir: str = None, autoshow=False):
        if output_dir:
            output_dir = os.path.join(output_dir, str(self.label[0]))
        else:
            output_dir = str(self.label[0])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not measure:
            measure = self.measure

        patch_size_str = f"{self.block_shape[0]}x{self.block_shape[1]}"
        Image.fromarray(self.reconstructed_input,
                        'RGB').save(os.path.join(output_dir,
                                                 str(self.label[0]) +
                                                 f"_{measure}_{patch_size_str}_reconstructed.png"))
