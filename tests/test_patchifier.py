from unittest import TestCase

from deepclo.algorithms.algorithm_por import AlgorithmPOR, \
    rank_image_blocks as rib, sort_image_blocks
from deepclo.core.measures.measure_functions import map_measure_function, \
    Measure, Ordering
from deepclo.pipe.dataset import Dataset
from deepclo.utils import show_numpy_image
from PIL import Image
import numpy as np


class TestPatchifier(TestCase):
    dataset = Dataset(dataset_name='cifar10')
    dataset.unravel()
    # sample = dataset.x_train[0]
    label = dataset.y_train[0]
    sample = Image.open('cu.jpg')
    sample = sample.resize((1024, 1024))
    sample = np.asarray(sample)
    block_shape = (256, 256)
    patchifier = AlgorithmPOR(sample=sample, label=label, block_shape=block_shape)

    def test_split_image(self):
        self.patchifier.split_image()
        m_func = map_measure_function(Measure.ENTROPY)
        self.patchifier.save_image_blocks()

        blocks = self.patchifier.blocks
        ranks = rib(blocks, content_measure=Measure.ENTROPY)
        asc_blocks, asc_rank = sort_image_blocks(self.patchifier.blocks,
                                                 ranks, block_rank_ordering=0)
        assert asc_blocks.any(), asc_rank.any()
        dec_blocks, dec_rank = sort_image_blocks(self.patchifier.blocks,
                                                 ranks, block_rank_ordering=1)
        assert dec_blocks.any(), dec_rank.any()

    def test_construct_new_input(self):
        self.patchifier.split_image()
        new_input = self.patchifier.construct_new_input(Measure.ENTROPY, rank_order=0)
        self.patchifier.save_reconstructed_input(measure='Entropy')
        new_input_mi = self.patchifier.construct_new_input(Measure.MI, rank_order=0)
        self.patchifier.save_reconstructed_input(measure='MI')
        new_input_je = self.patchifier.construct_new_input(Measure.JE, rank_order=0)
        self.patchifier.save_reconstructed_input(measure='JE')
        assert not np.array_equal(new_input_mi, new_input)
        assert not np.array_equal(new_input_mi, new_input_je)
        assert not np.array_equal(new_input, new_input_je)
