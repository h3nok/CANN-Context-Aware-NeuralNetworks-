from abc import ABC, abstractmethod
import tensorflow as tf

SUPPORTED_DATASETS = {
    'keras':
        {
            'CIFAR10': tf.keras.datasets.cifar10,
            'CIFAR100': tf.keras.datasets.cifar100
        },
    'tf':
        {
            'IMAGENET2012': None,
            'CATS_VS_DOGS': None,
            'IMAGENET2012_SUBSET': None,
            'CALTECH101': None,
            'IMAGENET_RESIZED': None,
            'IMAGENET_RESIZED/32X32': None,
            'IMAGENET_RESIZED/64X64': None,
            'SHAPES': None
        }
}


class DatasetBase(ABC):
    def __init__(self, name: str = None):
        """
        Base interface to dataset providers

        Args:
            name:
        """
        supported_models = list(SUPPORTED_DATASETS['keras'].keys()) + list(SUPPORTED_DATASETS['tf'].keys())
        if not name or name == '' or name.upper() not in supported_models:
            raise RuntimeError(f"Supplied dataset name '{name}' doesn't exist. 'Must supply a valid dataset "
                               f"name from {','.join(supported_models)} ")

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
