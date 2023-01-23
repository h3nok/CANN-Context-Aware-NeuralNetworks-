import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import time
import tensorflow as tf
import matplotlib as mpl

COLORS = {
    'gold': '#cfb87c',
    'Train': '#cfb87c',
    'Test': '#000000',
    'black': '#000000',
    'dark_gray': '#565A5C',
    'blue': '#4b92db',
    'light_gray': '#A2A4A3',
    'brown': '#3e0f05',
    'yg': '#3e2b05',
    'gy': '#343e05',
    'yr': '#965906',
    'ry': '#d88009'
}

PLOTS_DIR = r'C:\Users\Henok\OneDrive\Research\Publications\Thesis\Plots'
plt.style.use('seaborn-darkgrid')
plt.rc('font', family='serif')


def show_images(images: np.ndarray, titles=None):
    if titles is None:
        titles = []

    from matplotlib import pyplot as plt

    number_of_images = images.shape[0]
    cols = int(np.log2(number_of_images))
    fig = plt.figure(figsize=(11., 11.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(cols, cols),  # creates a grid of axes
                     axes_pad=1,  # pad between axes in inch.
                     )
    if len(titles) == number_of_images:
        for ax, im, tit in zip(grid, images, titles):
            # Iterating over the grid returns the Axes.
            ax.imshow(im, interpolation='nearest', aspect='auto')
            ax.title.set_text(str(tit))
            ax.grid(False)
    else:
        for ax, im in zip(grid, images):
            # Iterating over the grid returns the Axes.
            ax.imshow(im, interpolation='nearest', aspect='auto')
            ax.grid(False)
        # Hide axes ticks

    plt.xticks([])
    plt.yticks([])

    return fig


def show_numpy_image(image: np.array):
    img = Image.fromarray(image, 'RGB')
    img.save('my.png')
    plt.imshow(image)
    img.show()


def configure_logger(module, logfile_dir='.', console=False):
    """
    Configure file logger - console is optional
    Args:
        module: name of the module - for log formatting
        logfile_dir: location where log file is to be stored
        console: set to true to print logs to console

    Returns: logger object

    """
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(logfile_dir, 'deep-clo.log'))
    fh.setLevel(logging.DEBUG)

    # create console handler
    ch = None
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)

    if console:
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def hist(data, title=None,
         x_label=None,
         label=None,
         bins='auto',
         color=COLORS['gold'],
         alpha=0.7, rwidth=0.85):
    """
    Histogram plotting routine

    Args:
        data:
        title:
        x_label:
        label:
        bins:
        color:
        alpha:
        rwidth:

    Returns:

    """
    assert isinstance(data, np.ndarray)
    n, bins, patches = plt.hist(x=data, label=label, bins=bins, color=color,
                                alpha=alpha, rwidth=rwidth, density=True)
    plt.grid(axis='y', alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


def scatter(x,
            y,
            title=None,
            x_label=None,
            y_label=None,
            alpha=0.75,
            save_as='plot.png',
            c=None):
    """

    Args:
        x:
        y:
        title:
        x_label:
        y_label:
        alpha:
        save_as:
        c:

    Returns:

    """
    save_as = os.path.join(PLOTS_DIR, save_as)
    area = y ** 2
    plt.scatter(x, y, s=area, c=c)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_as, dpi=300)


def multi_scatter(x, y, title=None, x_label=None, y_label=None, labels=None, alpha=0.70,
                  save_as='plot.png', c=None, loc='lower right'):
    assert len(x) == len(y) and len(x) > 1
    save_as = os.path.join(PLOTS_DIR, save_as)
    markers = ['o', '^']
    # area = y**2
    for i in range(len(x)):
        if c:
            plt.scatter(x[i], y[i], c=c[i], label=labels[i], marker=markers[i], s=4, alpha=alpha)
        else:
            plt.scatter(x[i], y[i], label=labels[i], marker=markers[i], s=4, alpha=alpha)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc=loc)
    plt.savefig(save_as, dpi=300)


def multi_hist(data, title, x_label, y_label, bins='auto', alpha=0.7, rwidth=0.85,
               save_as='plot.png'):
    assert isinstance(data, dict)
    for key, value in data.items():
        n, _, _ = plt.hist(value, bins=bins, alpha=alpha,
                           label=key, rwidth=rwidth, color=COLORS[key])

    plt.grid(axis='y', alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper left')
    if save_as:
        save_as = os.path.join(PLOTS_DIR, save_as)
        plt.savefig(save_as, dpi=300)
        plt.show()


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            print(sample)
            time.sleep(0.01)

    print("Execution time:", time.perf_counter() - start_time)


def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):
    # Remove invalid entries (negative times, or empty steps) from the timelines
    invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:, 0]
    steps = timeline['steps'][invalid_mask].numpy()
    times = timeline['times'][invalid_mask].numpy()
    values = timeline['values'][invalid_mask].numpy()

    # Get a set of different steps, ordered by the first time they are encountered
    step_ids, indices = np.stack(np.unique(steps, return_index=True))
    step_ids = step_ids[np.argsort(indices)]

    # Shift the starting time to 0 and compute the maximal time value
    min_time = times[:, 0].min()
    times[:, 0] = (times[:, 0] - min_time)
    end = max(width, (times[:, 0] + times[:, 1]).max() + 0.01)

    cmap = mpl.cm.get_cmap("plasma")
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title)
    fig.set_size_inches(17.0, len(step_ids))
    plt.xlim(-0.01, end)

    for i, step in enumerate(step_ids):
        step_name = step.decode()
        ax = axs[i]
        ax.set_ylabel(step_name)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_xticklabels([])
        ax.grid(which="both", axis="x", color="k", linestyle=":")

        # Get timings and annotation for the given step
        entries_mask = np.squeeze(steps == step)
        series = np.unique(times[entries_mask], axis=0)
        annotations = values[entries_mask]

        ax.broken_barh(series, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(series):
                annotation = "\n".join([f"{l}: {v}" for l, v in zip(("i", "e", "s"), annotations[j])])
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")
