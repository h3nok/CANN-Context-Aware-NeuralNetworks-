from unittest import TestCase

from deepclo.algorithms.curriculum import Curriculum
from deepclo.algorithms.por import POR
from deepclo.core.measures.measure_functions import Measure
from deepclo.pipe.dataset import DeepCLODataProvider


class TestDataset(TestCase):
    ds = DeepCLODataProvider(dataset_name='cifar10')
    assert ds.x_train.shape == (50000, 32, 32, 3)
    assert ds.input_shape == (32, 32, 3)

    def test_plot_distributions(self):
        self.ds.plot_dataset_measure_distribution(limit=-1)
        self.ds.plot_dataset_measure_distribution(Measure.MI, limit=10)
        self.ds.plot_dataset_measure_distribution(Measure.JE, limit=10)
        self.ds.plot_dataset_measure_distribution(Measure.CROSS_ENTROPY, limit=10)

    def test_por_provider(self):
        por = POR()
        por.block_shape = (8, 8)
        por.rank_order = 0
        por.measure = 'Entropy'

        # new_input, label = por.preprocess_input(sample=self.sample, label=self.label)
        prefetched = self.ds.train_dataset(batch_size=4, train_preprocessing=por.algorithm_por)

        iterator = iter(prefetched)
        batches = iterator.next()

        while batches:
            for sample in batch:

                print(sample.shape)

            batch = iterator.next()

    def test_clo_provider(self):
        curr = Curriculum()
        curr.measure = 'Entropy'
        curr.rank_order = 0
        curr.reference_image_index = 0

        dataset = self.ds.train_dataset(batch_size=8,
                                        train_preprocessing=curr.generate_syllabus,
                                        clo=True)
        iterator = iter(dataset)
        batches = iterator.next()
        print(batches[0].shape)
