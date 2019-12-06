from dataset import DataSet
from trainingsetstat import TrainingSetStat
import plots
import numpy as np
import matplotlib.pyplot as plt
from research_datasets import ImageDatasets

if __name__ == '__main__':
    ds = 'cifar10'
    cifar10_ds = DataSet(name=ImageDatasets.caltech101.value)
    total_samples = cifar10_ds.info.splits['test']
    ts_stat = TrainingSetStat(cifar10_ds)
    data = ts_stat.entropy('train')
    entropies = np.array([x[1] for x in list(data.values())])
    labels = np.array([x[0] for x in list(data.values())])

    plots.hist(data=entropies,
               title='Entropy distribution of {}'.format(ds.upper()),
               x_label='Entropy')

    plt.show()
    plt.close()
    plots.scatter(labels, entropies,
                  title='Entropy Distribution of {} across Categories'.format(ds.upper()),
                  x_label='Category', y_label='Entropy')
    plt.show()
