from datasetstat import DatasetStat
from dataset import Dataset
from research_datasets import ImageDatasets
import plots


class ImageNet(Dataset):
    def __init__(self):
        super().__init__(name=ImageDatasets.imagenet2012.value)
        self._dataset = Dataset(name=ImageDatasets.imagenet2012.value)
        assert self._dataset

        self._stats = DatasetStat(self._dataset)
        self._entropies_train, _labels_train = self._stats.entropy('train')
        self._entropies_test, _labels_test = self._stats.entropy('test')

    def plot_entropy_histograms(self, colors=None):
        plots.multi_hist([self._entropies_train, self._entropies_test],
                         labels=['train', 'test'],
                         title="Entropy distribution of {}".format(ImageDatasets.imagenet2012.value),
                         x_label='Entropy', y_label='Count',
                         bins=1000, alpha=0.75, colors=colors,
                         save_as=ImageDatasets.imagenet2012.value)
