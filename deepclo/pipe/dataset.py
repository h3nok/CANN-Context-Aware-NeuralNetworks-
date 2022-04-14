import tensorflow as tf
from matplotlib import pyplot

from deepclo.algorithms.image_processsing import assess_and_rank_images
from deepclo.core.measures.measure_functions import Measure
from deepclo.pipe.dataset_base import DatasetBase, SUPPORTED_DATASETS
from deepclo.utils import multi_hist


class DeepCLODataProvider(DatasetBase):

    def __init__(self, dataset_name: str = None):
        """
        Interface to construct image classification datasets using POR-enabled
        os syllabus pipeline

        Args:
            dataset_name: name of dataset to load
        """
        super().__init__(dataset_name)
        self.x_train, self.y_train = None, None
        self.x_test, y_test = None, None

        self._train_preprocessing = None
        self._test_preprocessing = None

        self._load()
        tf.data.experimental.enable_debug_mode()

        self._train_ds = None
        self._validation_ds = None

        assert self.dataset

    def _load(self):
        """
        Load dataset, from keras

        Returns: keras dataset
        TODO - add support for tfds
        """

        if self._name.upper() in SUPPORTED_DATASETS.keys():
            self.dataset = SUPPORTED_DATASETS[self._name.upper()].load_data()
        else:
            print(list(SUPPORTED_DATASETS.keys()))
            raise RuntimeError("Supplied dataset name is not supported. "
                               "Please select one from the list")

        self._unravel()

    def _unravel(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset

    @property
    def input_shape(self):
        return self.x_train.shape[1:]

    def train_dataset(self, batch_size, train_preprocessing=None, clo=False):
        """
        Training data provider

        Args:
            batch_size:
            train_preprocessing:
            clo:

        Returns: tf.data

        """
        self._train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))

        if train_preprocessing:
            if clo:
                self._train_ds = self._train_ds.batch(batch_size)
                self._train_ds = self._train_ds.map(
                    lambda x, y: (train_preprocessing(x, y)),
                    num_parallel_calls=tf.data.AUTOTUNE)

                return self._train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

            else:
                self._train_ds = self._train_ds.map(
                    lambda x, y: (train_preprocessing(x, y)),
                    num_parallel_calls=tf.data.AUTOTUNE)

                self._train_ds = self._train_ds.batch(batch_size)

                return self._train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return (
            self._train_ds.shuffle(len(self.x_train))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
        )

    def test_dataset(self, batch_size, test_preprocessing=None):
        """
        Test data provider

        Args:
            batch_size:
            test_preprocessing:

        Returns: tf.data

        """
        self._validation_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

        if test_preprocessing:
            return (
                self._validation_ds.shuffle(len(self.x_test))
                    .map(test_preprocessing)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE)
            )

        return (
            self._validation_ds.shuffle(len(self.x_test))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
        )

    def plot_images(self, limit=9, dataset_type='Train'):
        """
        Plot a few images from the dataset

        Args:
            limit: number of samples to plot
            dataset_type: train or test dataset

        Returns:

        """

        if dataset_type == 'Train':
            dataset = self.x_train
        else:
            dataset_type = 'Test'
            dataset = self.x_test

        for i in range(limit):
            pyplot.subplot(330 + 1 + i)
            pyplot.imshow(dataset[i])

        pyplot.title(f"{self.dataset} - {dataset_type}")

        pyplot.show()

    def plot_dataset_measure_distribution(self,
                                          measure=Measure.ENTROPY,
                                          limit=100,
                                          reference_block_index=0):
        """
        Plot distribution of content measure (similarity) of the dataset

        Args:
            reference_block_index:
            measure:
            limit:

        Returns:

        """
        data = {}
        train_ranks = assess_and_rank_images(self.x_train[:limit],
                                             content_measure=measure,
                                             reference_block_index=reference_block_index)

        val_ranks = assess_and_rank_images(self.x_test[:limit],
                                           content_measure=measure,
                                           reference_block_index=reference_block_index)

        data['Train'] = train_ranks
        data['Test'] = val_ranks

        measure_str = str(measure).split('.')[1]
        title = f"{self.name.upper()} {measure_str} distribution"
        save_as = f"{self.name}_{measure}.png"

        multi_hist(data, title=title, x_label=measure_str, y_label='Frequency', save_as=save_as)

    def __repr__(self):
        return f""" {self.__class__.__name__}
            \tName   : {self.name}
            \tTrain  : {self.x_train.shape}
            \tTest   : {self.x_test.shape}
        """
