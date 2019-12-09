from cifar import Cifar
from caltech import Caltech
from imagenet import ImageNet
from research_datasets import Custom_Color
import matplotlib.pyplot as plt
from map_measure import Measure

if __name__ == '__main__':
    # ds = Cifar(metric=Measure.IQ)
    c = [Custom_Color['gold'], Custom_Color['black']]
    # ds.imq_train_test_scatter_plot(colors=c, dataset='cifar100')
    # ds.imq_train_test_scatter_plot(colors=c, dataset='cifar10')
    # ds.plot_cifar100_im_q_histograms(colors=c)
    # ds.plot_cifar10_im_q_histograms(colors=c)
    # ds = Cifar(metric=Measure.ENTROPY)
    # ds.entropy_train_test_scatter_plot(colors=c, dataset='cifar10')
    # ds.entropy_train_test_scatter_plot(colors=c, dataset='cifar100')
    # ds.plot_cifar100_entropy_histograms(colors=c)
    # ds.plot_cifar10_entropy_histograms(colors=c)
    caltech = Caltech()
    caltech.plot_caltech101_entropy_histograms(colors=c)
    caltech.entropy_train_test_scatter_plot(colors=c)
    # # plt.show()
