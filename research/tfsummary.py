from enum import Enum
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from scipy.interpolate import spline, make_interp_spline, BSpline
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class InceptionV2Loss(Enum):
    regularization = "regularization_loss_1"
    total = "total_loss_1"
    softmax = "losses/softmax_cross_entropy_loss/value"
    mixed_4c_sparsity = 'sparsity/Mixed_4c'
    mixed_3b_sparsity = 'sparsity/Mixed_3b'
    mixed_5b_sparsity = 'sparsity/Mixed_5b'
    learning_rate = 'learning_rate'


METRIC_FULL_NAME = {
    'ce': "Conditional Entropy",
    'mn': "Max Norm",
    'ssim': "Structural Similarity Index",
    'psnr': 'Peak-Signal-to-Noise-Ratio',
    'min': 'Normalized Mutual Information',
    'mi': 'Mutual Information',
    'e': "Entropy",
    'kl': "Kullback-Leibler Divergence",
    'l2': 'L2-Norm',
    'l1': 'L1-Norm',
    'je': 'Joint Entropy',
    'iv': 'Information Variation',
    'baseline': 'Baseline (no-curriculum)',
    'base': 'Baseline (no-curriculum)',
    'cross_entropy': 'Cross Entropy',
    'cep': 'Cross Entropy PMF'
}


class InceptionNetSummary(object):
    event_file = ""
    tags = []

    def __init__(self, event_file_path):
        assert event_file_path is not None, "Must supply a valid events file\n"
        self.event_file = event_file_path

    def process_sparsity(self):
        event_acc = EventAccumulator(self.event_file)
        event_acc.Reload()
        self.tags = event_acc.Tags()
        # import pprint
        # pprint.pprint(event_acc.Tags())

        mixed_4c_summary = event_acc.Scalars(InceptionV2Loss.mixed_4c_sparsity.value)
        mixed_3b_summary = event_acc.Scalars(InceptionV2Loss.mixed_3b_sparsity.value)
        mixed_5b_summary = event_acc.Scalars(InceptionV2Loss.mixed_5b_sparsity.value)
        learning_rate_summary = event_acc.Scalars(InceptionV2Loss.learning_rate.value)

        steps = []
        mixed_4c = []
        mixed_3b = []
        mixed_5b = []
        walltime = []
        learning_rate_decay = []

        for item in mixed_4c_summary:
            walltime.append(item[0])
            steps.append(item[1])
            mixed_4c.append(item[2])
        for item in mixed_3b_summary:
            mixed_3b.append(item[2])
        for item in mixed_5b_summary:
            mixed_5b.append(item[2])
        for item in learning_rate_summary:
            learning_rate_decay.append(item[2])

        return steps, mixed_4c, mixed_3b, mixed_5b, learning_rate_decay, walltime

    def process_loss(self):
        event_acc = EventAccumulator(self.event_file)
        event_acc.Reload()
        self.tags = event_acc.Tags()
        # import pprint
        # pprint.pprint(event_acc.Tags())

        total_loss_summary = event_acc.Scalars(InceptionV2Loss.total.value)
        regularization_loss_summary = event_acc.Scalars(InceptionV2Loss.regularization.value)
        softmax_loss_summary = event_acc.Scalars(InceptionV2Loss.softmax.value)

        wall_time = []
        steps = []
        total_loss = []
        reg_loss = []
        softmax_loss = []

        for item in total_loss_summary:
            wall_time.append(item[0])
            steps.append(item[1])
            total_loss.append(item[2])

        for item in regularization_loss_summary:
            reg_loss.append(item[2])

        for item in softmax_loss_summary:
            softmax_loss.append(item[2])

        return steps, total_loss, reg_loss, softmax_loss


def _plot_multiple(data, plot='Softmax loss', network='Inception V2', outdir=".", format="png"):
    for run in data:
        x = np.asarray(run['Steps'])
        xnew = np.linspace(x.min(), x.max(), 300)
        y = np.asarray(run[plot])
        spl = make_interp_spline(x, y, k=3)
        ynew = spl(xnew)
        plt.plot(xnew, ynew, label=run['Metric'], c=np.random.rand(3))

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("{} {}".format(network, plot))
    plt.legend(loc='upper right', frameon=False, fontsize='small')
    plt.savefig(os.path.join(outdir, '{}_{}.{}'.format(plot, network, format)), format=format, dpi=1000)
    plt.show()
    plt.close()


def moving_average(data, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')


def _plot_sparsity(data, plot='Mixed 4c', output_dir='.', format='png'):
    data = data[1]
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Cook the data
    X = data['Sparsity steps']
    Y = data[plot]
    X, Y = np.meshgrid(X, Y)

    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # customize z axis
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_losses(data, output_dir):
    _plot_multiple(data, outdir=output_dir)
    _plot_multiple(data, plot="Total loss", outdir=output_dir)
    _plot_multiple(data, plot="Regularization loss", outdir=output_dir)
    _plot_multiple(data, plot="Learning rate decay", outdir=output_dir)


def run(summaries_dir="C:\\data\\tensorboard\\curriculum\\Train\\inception_v1\\cifar10",
        output_dir=".", iterations=100000):
    training_data = []
    data = {}
    for measure in os.listdir(summaries_dir):
        metric = measure
        event_file_dir = os.path.join(summaries_dir, metric, str(iterations))
        if not os.path.exists(event_file_dir): continue
        files = os.listdir(event_file_dir)
        event_file = None
        for file in files:
            if file.endswith('.node18'):
                event_file = file
        if not event_file: continue
        log_file = os.path.join(event_file_dir, event_file)
        s = InceptionNetSummary(log_file)
        steps, loss, reg_loss, softmax_loss = s.process_loss()
        sparsity_steps, mixed_4c, mixed_3b, mixed_5b, lr, walltime = s.process_sparsity()
        data = {
            "Steps": steps,
            "Total loss": loss,
            "Metric": METRIC_FULL_NAME[metric],
            "Regularization loss": reg_loss,
            "Learning rate decay": lr,
            "Softmax loss": softmax_loss,
            'Sparsity steps': sparsity_steps,
            'Mixed 4c': mixed_4c,
            'Mixed 3b': mixed_3b,
            'Mixed 5b': mixed_5b,
            'Wall time': walltime
        }
        training_data.append(data)

    plot_losses(training_data, output_dir=output_dir)


if __name__ == "__main__":
    run()
