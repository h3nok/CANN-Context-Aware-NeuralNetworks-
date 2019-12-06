from training_sample import Sample


class TrainingSetStat:
    _training_set = None
    _entropies = {'test': {},
                  'train': {}}
    _train = None
    _test = None

    def __init__(self, training_set):
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
                entropies[label_str] = ts.entropy
                counter += 1
            self._entropies[split] = entropies

    def entropy(self, split='train'):
        assert split in ['train', 'test']
        if len(self._entropies[split]) > 0:
            return self._entropies[split]
        else:
            self._entropy(split)
            return self._entropies[split]
