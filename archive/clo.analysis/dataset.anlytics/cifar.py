from dataset import Dataset
from datasetstat import DatasetStat
import plots
import matplotlib.pyplot as plt
from research_datasets import ImageDatasets
from map_measure import Measure


class Cifar:
    def __init__(self, metric=Measure.ENTROPY):
        self._cifar_10 = Dataset(name=ImageDatasets.dataset.value)
        self._cifar_100 = Dataset(name=ImageDatasets.cifar100.value)
        print(self._cifar_10.info)
        print(self._cifar_100.info)
        assert self._cifar_10
        assert self._cifar_100

        self._cifar10_stats = DatasetStat(self._cifar_10)
        self._cifar100_stats = DatasetStat(self._cifar_100)

        self._cifar10_entropies_train, self._cifar10_labels_train = self._cifar10_stats.entropy('train')
        print(len(self._cifar10_entropies_train))
        print(len(self._cifar10_labels_train))

        self._cifar10_entropies_test, self._cifar10_labels_test = self._cifar10_stats.entropy('test')
        self._cifar100_entropies_train, self._cifar100_labels_train = self._cifar100_stats.entropy('train')
        self._cifar100_entropies_test, self._cifar100_labels_test = self._cifar100_stats.entropy('test')

        # self._cifar10_fft_iq_train, self._cifar10_labels_train = self._cifar10_stats.quality_measure('train')
        # self._cifar10_fft_iq_test, self._cifar10_labels_test = self._cifar10_stats.quality_measure('test')
        # self._cifar100_fft_iq_train, self._cifar100_labels_train = self._cifar100_stats.quality_measure('train')
        # self._cifar100_fft_iq_test, self._cifar100_labels_test = self._cifar100_stats.quality_measure('test')

        # assert len(set(self._cifar100_labels_train)) == self._cifar_100.num_classes

    def plot_cifar10_entropy_histograms(self, colors=None):
        plt.close()
        plots.multi_hist([self._cifar10_entropies_train, self._cifar10_entropies_test],
                         labels=['train', 'test'],
                         title="Entropy distribution of {}".format(ImageDatasets.dataset.value),
                         x_label='Entropy', y_label='Count',
                         bins=1000, alpha=0.75, colors=colors,
                         save_as='cifar10_entropy_hist.png')

    def plot_cifar100_entropy_histograms(self, colors=None):
        plt.close()
        plots.multi_hist([self._cifar100_entropies_train, self._cifar100_entropies_test],
                         labels=['train', 'test'],
                         title="Entropy distribution of {}".format(ImageDatasets.cifar100.value),
                         x_label='Entropy', y_label='Count',
                         bins=1000, alpha=0.75, colors=colors, save_as='cifar100_entropy_hist.png')

    def plot_cifar100_im_q_histograms(self, colors=None):
        plt.close()
        plots.multi_hist([self._cifar100_fft_iq_train, self._cifar100_fft_iq_test],
                         labels=['train', 'test'],
                         title="Quality measure distribution of {}".format(ImageDatasets.cifar100.value),
                         x_label='Entropy', y_label='Count',
                         bins=1000, alpha=0.75, colors=colors, save_as='cifar100_fft_iq_hist.png')

    def plot_cifar10_im_q_histograms(self, colors=None):
        plt.close()
        plots.multi_hist([self._cifar10_fft_iq_train, self._cifar10_fft_iq_test],
                         labels=['train', 'test'],
                         title="Quality measure distribution of {}".format(ImageDatasets.dataset.value),
                         x_label='Entropy', y_label='Count',
                         bins=1000, alpha=0.75, colors=colors, save_as='cifar100_fft_iq_hist.png')

    def entropy_train_test_scatter_plot(self, dataset=None, colors=None):
        plt.close()
        if dataset == ImageDatasets.dataset.value:
            plots.multi_scatter([self._cifar10_labels_train, self._cifar10_labels_test],
                                [self._cifar10_entropies_train, self._cifar10_entropies_test],
                                title="Entropy distribution of {} across labels".format(ImageDatasets.dataset.value),
                                c=colors, labels=['train', 'test'], x_label="Classes", y_label='Entropy',
                                save_as=ImageDatasets.dataset.value + '_entropy_scatter.png')
        elif dataset == ImageDatasets.cifar100.value:
            plots.multi_scatter([self._cifar100_labels_train, self._cifar100_labels_test],
                                [self._cifar100_entropies_train, self._cifar100_entropies_test],
                                title="Entropy distribution of {} across labels".format(ImageDatasets.cifar100.value),
                                c=colors, labels=['train', 'test'], x_label='Classes', y_label='Entropy',
                                save_as=ImageDatasets.cifar100.value + '_entropy_scatter.png')

    def imq_train_test_scatter_plot(self, dataset=None, colors=None):
        plt.close()
        if dataset == ImageDatasets.dataset.value:
            plots.multi_scatter([self._cifar10_labels_train, self._cifar10_labels_test],
                                [self._cifar10_fft_iq_train, self._cifar10_fft_iq_test],
                                title="Image Quality distribution of {} across labels".format(ImageDatasets.dataset.value),
                                c=colors, labels=['train', 'test'], x_label="Classes", y_label='Image Quality',
                                save_as=ImageDatasets.dataset.value + '_imq_scatter.png', loc='upper right')
        elif dataset == ImageDatasets.cifar100.value:
            plots.multi_scatter([self._cifar100_labels_train, self._cifar100_labels_test],
                                [self._cifar100_fft_iq_train, self._cifar100_fft_iq_test],
                                title="Image Quality distribution of {} across labels".format(ImageDatasets.cifar100.value),
                                c=colors, labels=['train', 'test'], x_label='Classes', y_label='Image Quality',
                                save_as=ImageDatasets.cifar100.value + '_imq_scatter.png', loc='upper right')

    def to_tfrecord(self, dataset='cifar10'):
        if dataset == 'cifar10':
            self._cifar_10.ds2tfrecord('train', r'E:\Thesis\Datasets\cifar10\\' + 'cifar10_train.tfrecord')
            self._cifar_10.ds2tfrecord('test', r'E:\Thesis\Datasets\cifar10\\' + 'cifar10_test.tfrecord')

        elif dataset == 'cifar100':
            self._cifar_100.ds2tfrecord('test', r'E:\Thesis\Datasets\cifar100\\' + 'cifar100_test.tfrecord')
            self._cifar_100.ds2tfrecord('train', r'E:\Thesis\Datasets\cifar100\\' + 'cifar100_train.tfrecord')
