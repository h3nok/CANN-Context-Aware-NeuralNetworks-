from cifar import Cifar
from imagenet import ImageNet
from research_datasets import Custom_Color
from map_measure import Measure
import matplotlib.pyplot as plt


def harmonic_mean(precision, recall):
    return (2*precision*recall)/(precision + recall)


if __name__ == '__main__':

    ds = Cifar(metric=Measure.ENTROPY)
    # ds.to_tfrecord('cifar10')
    c = [Custom_Color['gold'], Custom_Color['black']]
    ds.entropy_train_test_scatter_plot(dataset='cifar10', colors=c)
    # ds.plot_cifar10_entropy_histograms()
    # ds.imq_train_test_scatter_plot(colors=c, pipe='Cifar10')
    # ds.imq_train_test_scatter_plot(colors=c, pipe='cifar10')
    # ds.plot_cifar100_im_q_histograms(colors=c)
    #
    # ds.plot_cifar10_im_q_histograms(colors=c)
    # ds = Cifar(metric=Measure.ENTROPY)
    # ds.entropy_train_test_scatter_plot(colors=c, pipe='cifar10')
    # ds.entropy_train_test_scatter_plot(colors=c, pipe='cifar100')
    # ds.plot_cifar100_entropy_histograms(colors=c)
    # ds.plot_cifar10_entropy_histograms(colors=c)
    # caltech = Caltech()
    # caltech.plot_caltech101_entropy_histograms(colors=c)
    # caltech.entropy_train_test_scatter_plot(colors=c)
    plt.show()
