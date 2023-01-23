import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot

from deepclo.algorithms.image_processsing import assess_and_rank_images
from deepclo.core.measures.measure_functions import Measure
from deepclo.pipe.dataset_base import DatasetBase, SUPPORTED_DATASETS
from deepclo.utils import multi_hist
from experiments.synthetic.synthetic_data import ShapesDataset
import os
import keras


class ImageDataProvider(DatasetBase):

    def __init__(self,
                 dataset_name: str = None,
                 train_limit: int = None,
                 val_limit: int = None, custom_dataset_path=None):
        """
        Interface to construct image classification datasets using POR-enabled
        os syllabus pipeline

        Args:
            dataset_name: name of dataset to load
        """

        super().__init__(dataset_name)

        self.val_limit = val_limit
        self.train_limit = train_limit
        self.x_train, self.y_train = [], []
        self.x_test, self.y_test = [], []

        self.custom_dataset_path = custom_dataset_path
        self._train_preprocessing = None
        self._test_preprocessing = None
        self._train_ds = None
        self._val_ds = None
        self._input_shape = (32, 32, 3)
        self.classes = set()
        tf.data.experimental.enable_debug_mode()

        self._load()

    def _load(self):
        """
        Load dataset, from keras - Returns: keras dataset

        TODO - add support for tfds
        """

        if self._name.upper() in list(SUPPORTED_DATASETS['keras'].keys()):
            self.dataset = SUPPORTED_DATASETS['keras'][self._name.upper()].load_data()
            self._unravel()
            self._input_shape = self.x_train.shape[1:]

        elif 'SHAPES' in self._name.upper():
            assert os.path.exists(self.custom_dataset_path)
            self.dataset = ShapesDataset(dataset_path=self.custom_dataset_path)
            self._unravel()

        elif self._name.upper() in SUPPORTED_DATASETS['tf'].keys():
            if self._name == 'cats_vs_dogs':
                split = ['train[:80%]', 'train[80%:]']
                self._train_ds, self._val_ds = tfds.as_numpy(tfds.load(name=self._name,
                                                                       split=split,
                                                                       shuffle_files=False,
                                                                       as_supervised=True)
                                                             )

            else:
                self._train_ds = tfds.as_numpy(tfds.load(name=self._name,
                                                         split='train',
                                                         shuffle_files=False,
                                                         as_supervised=True)
                                               )
                self._val_ds = tfds.as_numpy(tfds.load(name=self._name,
                                                       split='validation',
                                                       shuffle_files=False,
                                                       as_supervised=True)
                                             )
            self._unravel()

        else:
            print(list(SUPPORTED_DATASETS.keys()))
            raise RuntimeError(f"Supplied dataset name '{self._name}' is not supported. "
                               "Please select one from the list")

    def _unravel(self):
        if self._name.upper() in SUPPORTED_DATASETS['keras'].keys():
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset
            self.y_train = self.y_train.astype(np.uint8)
            self.y_test = self.y_test.astype(np.uint8)

        elif 'SHAPES' in self._name.upper():
            (self.x_train, self.y_train) = self.dataset.train_dataset
            (self.x_test, self.y_test) = self.dataset.test_dataset

        else:
            for index, sample in enumerate(self._val_ds):
                self.x_test.append(np.resize(sample[0], self._input_shape).astype(np.uint8))
                self.y_test.append(np.array([sample[1]]).astype(np.uint8))
                self.classes.add(sample[1])
                if index % 1000 == 0:
                    print(f"Processed {index} val samples ... ")

            print(f"Successfully processed validation dataset, "
                  f"train: {len(self.x_test)}, classes: {len(self.classes)}")

            for index, sample in enumerate(self._train_ds):
                self.x_train.append(np.array(sample[0]).astype(np.uint8))
                self.y_train.append(np.array([sample[1]]).astype(np.uint8))

                if self.train_limit:
                    if index > self.train_limit:
                        break

                if index % 1000 == 0:
                    print(f"Processed {index} train samples ... ")

        self.y_train = keras.utils.np_utils.to_categorical(self.y_train, self.num_classes, dtype=np.uint8)
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test, self.num_classes, dtype=np.uint8)

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def num_classes(self):
        return len(self.classes)

    def train_dataset(self, batch_size, train_preprocessing=None, clo=False) -> tf.data:
        """
        Training data provider -

        Args:
            batch_size: int - batch size
            train_preprocessing: - preprocessing algorithm function
            clo: bool - true if CLO is applied

        Returns: tf.data

        """
        # if self._name.upper() in SUPPORTED_DATASETS['keras'].keys():
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

    def test_dataset(self, batch_size, test_preprocessing=None) -> tf.data:
        """
        Test data provider -

        Args:
            batch_size: int - batch size
            test_preprocessing: preprocessing algorithm function

        Returns: tf.data
        """

        self._val_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))

        if test_preprocessing:
            return (
                self._val_ds.shuffle(len(self.x_test))
                .map(test_preprocessing)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        return (
            self._val_ds.shuffle(len(self.x_test))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    def plot_images(self, limit=9, dataset_type='Train') -> None:
        """
        Plot a few images from the dataset -

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
                                          reference_block_index=0) -> None:
        """
        Plot distribution of content measure (similarity) of the dataset

        Args:
            reference_block_index: reference block if measure is a distance measure.
            measure: content assessment measure
            limit: int - number of samples to assess

        Returns: None
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
            \tInput shape : {self.input_shape}
            \tClasses     : {self.num_classes}
        """
