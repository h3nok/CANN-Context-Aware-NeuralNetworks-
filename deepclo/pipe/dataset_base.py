from abc import ABC, abstractmethod


SUPPORTED_DATASETS = ['catsanddogs', 'cifar10', 'cifar100', ]


class DatasetBase(ABC):
    def __init__(self, name: str = None):
        if not name or name == '':
            raise RuntimeError(f"Must supply a valid dataset name from {','.join(SUPPORTED_DATASETS)} ")

        self._name = name
        self.dataset = None

    @property
    def name(self):
        return self._name

    @abstractmethod
    def _build(self):
        raise NotImplementedError("Derived class must implement a dataset getter function")

    @abstractmethod
    def unravel(self):
        raise NotImplementedError("Derived class must implement a dataset getter function")