from abc import ABC, abstractmethod
import tensorflow as tf

SUPPORTED_DATASETS = {
    'CIFAR10': tf.keras.datasets.cifar10,
    'CIFAR100': tf.keras.datasets.cifar100,
}


class DatasetBase(ABC):
    def __init__(self, name: str = None):
        if not name or name == '' or name.upper() not in list(SUPPORTED_DATASETS.keys()):
            raise RuntimeError(f"Must supply a valid dataset "
                               f"name from {','.join(list(SUPPORTED_DATASETS.keys()))} ")

        self._name = name

        self.dataset = None

    @property
    def name(self):
        return self._name

    @abstractmethod
    def _load(self):
        raise NotImplementedError("Derived class must implement a dataset getter function")

    @abstractmethod
    def _unravel(self):
        raise NotImplementedError("Derived class must implement a dataset getter function")
