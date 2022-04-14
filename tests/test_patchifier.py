from unittest import TestCase

import numpy as np
from PIL import Image

from deepclo.algorithms.por import POR, \
    assess_and_rank_images as rib, sort_images
from deepclo.core.measures.measure_functions import map_measure_function, \
    Measure
from deepclo.pipe.dataset import DeepCLODataProvider
from deepclo.utils import show_numpy_image


class TestPatchifier(TestCase):
    label = np.array([1])
    sample = Image.open('cu.jpg')
    sample = sample.resize((1024, 1024))
    sample = np.asarray(sample)
    block_shape = (256, 256)
    patchifier = POR(sample=sample, label=label, block_shape=block_shape)

    def test_split_image(self):
        self.patchifier.split_image(block_shape=self.block_shape)
        m_func = map_measure_function(Measure.ENTROPY)
        self.patchifier.save_image_blocks()

        blocks = self.patchifier.blocks
        ranks = rib(blocks, content_measure=Measure.ENTROPY)
        asc_blocks, asc_rank = sort_images(self.patchifier.blocks,
                                           ranks, block_rank_ordering=0)
        assert asc_blocks.any(), asc_rank.any()
        dec_blocks, dec_rank = sort_images(self.patchifier.blocks,
                                           ranks, block_rank_ordering=1)
        assert dec_blocks.any(), dec_rank.any()

    def test_construct_new_input(self):
        self.patchifier.split_image()
        new_input = self.patchifier.construct_new_input_from_blocks()
        # self.patchifier.save_reconstructed_input(measure='Entropy')
        # new_input_mi = self.patchifier.construct_new_input_from_blocks(Measure.MI, rank_order=0)
        # self.patchifier.save_reconstructed_input(measure='MI')
        # new_input_je = self.patchifier.construct_new_input_from_blocks(Measure.JE, rank_order=0)
        # self.patchifier.save_reconstructed_input(measure='JE')
        #
        # assert not np.array_equal(new_input_mi, new_input)
        # assert not np.array_equal(new_input_mi, new_input_je)
        # assert not np.array_equal(new_input, new_input_je)

    def test_preprocess(self):
        dataset = DeepCLODataProvider(dataset_name='cifar10')
        dataset._unravel()
        label = dataset.y_train[0]
        por = POR()
        por.block_shape = (512, 512)
        por.rank_order = 0
        por.measure = 'Entropy'
        new_input, label = por.preprocess_input(sample=self.sample, label=self.label)
        show_numpy_image(new_input)
