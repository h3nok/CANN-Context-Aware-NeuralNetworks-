from training_sample import Sample
import numpy as np
import uuid
import csv
import operator


class DatasetStat:
    _training_set = None
    _entropies = {'test': [], 'train': []}
    _fft_iqs = {'test': [], 'train': []}

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
                entropies[counter] = (uuid.uuid1(), label_str, ts.entropy)
                counter += 1

            self._entropies[split] = sorted(entropies.values(), key=operator.itemgetter(2))

        counter = 0
        if split == 'test':
            for sample in self._test:
                example, label = sample['image'], sample['label']
                label_str = self._training_set.labels[label]
                ts = Sample(example, label_str)
                entropies[counter] = (uuid.uuid1(), label_str, ts.entropy)
                counter += 1

            self._entropies[split] = sorted(entropies.values(), key=operator.itemgetter(2))

    def _fft_iq(self, split='train'):
        ffts = {}
        assert split in ['train', 'test']
        counter = 0
        if split == 'train':
            for sample in self._train:
                example, label = sample['image'], sample['label']
                label_str = self._training_set.labels[label]
                ts = Sample(example, label_str)
                ffts[counter] = (uuid.uuid1(), label_str, ts.fft_iq)
                counter += 1

            self._fft_iqs[split] = sorted(ffts.values(), key=operator.itemgetter(2))

        counter = 0
        if split == 'test':
            for sample in self._test:
                example, label = sample['image'], sample['label']
                label_str = self._training_set.labels[label]
                unique_id = uuid.uuid1()
                ts = Sample(example, label_str, unique_id)
                ffts[counter] = (unique_id, label_str, ts.fft_iq)
                counter += 1

            self._fft_iqs[split] = sorted(ffts.values(), key=operator.itemgetter(2))

    def _mutual_information(self, split='train'):
        pass

    def entropy(self, split='train'):
        assert split in ['train', 'test']

        self._entropy(split)

        entropies = np.array([x[2] for x in self._entropies[split]])
        labels = np.array([x[1] for x in self._entropies[split]])
        ids = np.array([x[0] for x in self._entropies[split]])

        write_csv([ids, labels, entropies], self._training_set.name + "_entropy.csv")
        return entropies, labels

    def quality_measure(self, split='train'):
        assert split in ['train', 'test']

        self._fft_iq(split)

        entropies = np.array([x[2] for x in self._fft_iqs[split]])
        labels = np.array([x[1] for x in self._fft_iqs[split]])
        ids = np.array([x[0] for x in self._fft_iqs[split]])

        write_csv([ids, labels, entropies], save_as=self._training_set.name + "_image_quality.csv")

        return entropies, labels


def write_csv(items, save_as='dataset_stat.csv'):
    assert isinstance(items, list)
    assert len(items) == 3
    assert len(items[0]) > 1
    rows = zip(items[0], items[1], items[2])
    with open(save_as, "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        f.close()
