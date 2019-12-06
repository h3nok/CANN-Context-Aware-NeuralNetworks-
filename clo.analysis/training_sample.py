from ITT import itt
import numpy as np
import skimage


class Sample:
    _data = None
    _attributes = None
    _label = None

    def __init__(self, data, label):
        assert isinstance(data, np.ndarray)
        self._data = data
        self._label = label
        self._attributes = SampleStat(data)

    @property
    def entropy(self):
        return float(self._attributes.entropy())


class SampleStat:
    _sample = None

    def __init__(self, sample: np.ndarray):
        self._sample = sample

    def entropy(self):
        return round(skimage.measure.shannon_entropy(self._sample), 4)


