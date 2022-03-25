import tensorflow as tf
from deepclo.pipe.dataset_base import DatasetBase
from matplotlib import pyplot


class Dataset(DatasetBase):

    def __init__(self, dataset_name: str = None):
        super().__init__(dataset_name)
        self.x_train, self.y_train = None, None
        self.x_test, y_test = None, None

        self._build()

        assert self.dataset

    def _build(self):
        """

        Returns: keras dataset

        """
        if self._name.lower() == 'cifar10':
            self.dataset = tf.keras.datasets.cifar10.load_data()

        elif self._name.lower() == 'cifar100':
            self.dataset = tf.keras.datasets.cifar100.load_data()

        return self.dataset

    def unravel(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset

    def plot_images(self, limit=9, dataset_type='train'):
        if dataset_type == 'train':
            dataset = self.x_train
        else:
            dataset = self.y_train
        for i in range(limit):
            # define subplot
            pyplot.subplot(330 + 1 + i)
            # plot raw pixel data
            pyplot.imshow(dataset[i])
        # show the figure
        pyplot.show()


