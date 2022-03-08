from dataset import Dataset
from datasetstat import DatasetStat
from research_datasets import ImageDatasets, Custom_Color
import plots
import matplotlib.pyplot as plt


class Caltech:
    def __init__(self):
        self._caltech101 = Dataset(name=ImageDatasets.caltech101.value)
        print(self._caltech101.info)
        assert self._caltech101

        self._caltech101_stat = DatasetStat(self._caltech101)
        self._caltech101_entropies_test, self._test_labels = self._caltech101_stat.entropy('test')
        self._caltech101_entropies_train, self._train_labels = self._caltech101_stat.entropy('train')

    def plot_caltech101_entropy_histograms(self, colors=None):
        plt.close()
        plots.multi_hist([self._caltech101_entropies_test,
                          self._caltech101_entropies_train],
                         labels=['test', 'train'],
                         title="Entropy distribution of {}".format(ImageDatasets.caltech101.value),
                         x_label='Entropy', y_label='Count',
                         bins=1000, alpha=0.75, colors=colors,
                         save_as=ImageDatasets.caltech101.value + '.png')

    def plot_entropy_hist(self):
        plt.close()
        plots.hist(self._caltech101_entropies_train, x_label='Entropy',
                   y_label="Count", title="Entropy distribution  "
                                          "of {} training set".format(ImageDatasets.caltech101.value))

    def entropy_train_test_scatter_plot(self, colors=None):
        plt.close()
        plots.multi_scatter([self._train_labels, self._test_labels],
                            [self._caltech101_entropies_train, self._caltech101_entropies_test],
                            title="Entropy distribution of {} across labels".format(ImageDatasets.caltech101.value),
                            c=colors, labels=['train', 'test'], x_label="Classes", y_label='Entropy',
                            save_as=ImageDatasets.caltech101.value + '_entropy_scatter.png')
