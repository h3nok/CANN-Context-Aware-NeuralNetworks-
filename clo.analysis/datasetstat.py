from training_sample import Sample
import numpy as np


class DatasetStat:
    _training_set = None
    _entropies = {'test': {},
                  'train': {}}
    _fft_iqs = {'test': {},
                'train': {}}

    _train = None
    _test = None

    def __init__(self, training_set):
        assert training_set
        self._training_set = training_set
        self._train, self._test = self._training_set.to_numpy

    def _entropy(self, split='train'):
        entropies = {}
        assert split in ['train', 'test']
        counter = 0
        if split == 'train':
            for sample in self._train:
                example, label = sample['image'], sample['label']
                label_str = self._training_set.labels[label]
                ts = Sample(example, label_str)
                entropies[counter] = (label_str, ts.entropy)
                counter += 1

            self._entropies[split] = entropies

        counter = 0
        if split == 'test':
            for sample in self._test:
                example, label = sample['image'], sample['label']
                label_str = self._training_set.labels[label]
                ts = Sample(example, label_str)
                entropies[counter] = (label_str, ts.entropy)
                counter += 1

            self._entropies[split] = entropies

    def _fft_iq(self, split='train'):
        ffts = {}
        assert split in ['train', 'test']
        counter = 0
        if split == 'train':
            for sample in self._train:
                example, label = sample['image'], sample['label']
                label_str = self._training_set.labels[label]
                ts = Sample(example, label_str)
                ffts[counter] = (label_str, ts.fft_iq)
                counter += 1

            self._fft_iqs[split] = ffts

        counter = 0
        if split == 'test':
            for sample in self._test:
                example, label = sample['image'], sample['label']
                label_str = self._training_set.labels[label]
                ts = Sample(example, label_str)
                ffts[counter] = (label_str, ts.fft_iq)
                counter += 1

            self._fft_iqs[split] = ffts

    def _mutual_information(self, split='train'):
        reference_sample = None
        assert split in ['train', 'test']

    def entropy(self, split='train'):
        assert split in ['train', 'test']

        self._entropy(split)

        entropies = np.array([x[1] for x in list(self._entropies[split].values())])
        labels = np.array([x[0] for x in list(self._entropies[split].values())])

        return entropies, labels

    def quality_measure(self, split='train'):
        assert split in ['train', 'test']

        self._fft_iq(split)

        entropies = np.array([x[1] for x in list(self._fft_iqs[split].values())])
        labels = np.array([x[0] for x in list(self._fft_iqs[split].values())])

        return entropies, labels
