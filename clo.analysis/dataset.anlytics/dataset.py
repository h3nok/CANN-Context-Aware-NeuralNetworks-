import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    _train = None
    _test = None
    _loader = None
    _metadata = None
    _train_total = None
    _test_total = None

    def __init__(self, name=None, train=None, test=None, batch_size=None):
        assert name
        self._train = train
        self._test = test
        if not self._train or not self._test:
            if batch_size:
                self._loader, self._metadata = tfds.load(name, with_info=True, batch_size=batch_size)
            else:
                self._loader, self._metadata = tfds.load(name, with_info=True)
            self._train, self._test = self._loader['train'], self._loader['test']
            self._train_total = self._metadata.splits['train']
            self._test_total = self._metadata.splits['test']

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

    @property
    def total_training_samples(self):
        return self._train_total

    @property
    def total_test_samples(self):
        return self._test_total
