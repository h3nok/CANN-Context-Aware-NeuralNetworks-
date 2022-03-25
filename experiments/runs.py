from deepclo.pipe.dataset import Dataset
from deepclo.algorithms.algorithm_por import AlgorithmPOR

cifar10 = Dataset(dataset_name='cifar10')
# cifar100 = Dataset(dataset_name='cifar100')

cifar10.unravel()
cifar10.plot_images()
