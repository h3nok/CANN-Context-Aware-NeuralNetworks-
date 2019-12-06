import tensorflow_datasets as tfds
import tensorflow as tf
from enum import Enum


class DataSet:
    _train = None
    _test = None
    _loader = None
    _metadata = None

    def __init__(self, name='cifar10', train=None, test=None):
        assert name
        self._train = train
        self._test = test
        if not self._train or not self._test:
            self._loader, self._metadata = tfds.load(name, with_info=True)
            self._train, self._test = self._loader['train'], self._loader['test']

    def get(self):
        assert isinstance(self._train, tf.data.Dataset)
        assert isinstance(self._test, tf.data.Dataset)
        return self._train, self._test

    @property
    def features(self):
        return self._metadata.features

    @property
    def num_classes(self):
        return self._metadata.features['label'].num_classes

    @property
    def labels(self):
        return self._metadata.features['label'].names

    def show_examples(self, split='train'):
        fig = None
        if split == 'train':
            fig = tfds.show_examples(self._metadata, self._train)
        else:
            fig = tfds.show_examples(self._metadata, self._test)

        return fig

    @property
    def to_numpy(self):
        return tfds.as_numpy(self._train), tfds.as_numpy(self._test)

    @property
    def info(self):
        return self._metadata

