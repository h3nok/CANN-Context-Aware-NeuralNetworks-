import matplotlib.pyplot as plt
import numpy as np
import os

UCD_COLORS = {
    'gold': '#cfb87c',
    'black': '#000000',
    'dark_gray': '#565A5C',
    'blue': '#4b92db',
    'light_gray': '#A2A4A3'
}

PLOTS_DIR = r'E:\Thesis\OneDrive\Research\Publications\Deep Learning\2020\Dataset'
plt.style.use('seaborn-darkgrid')
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
# plt.style.use('dark_background')
# plt.style.use('dark_background')


def hist(data, title=None, x_label=None, label=None, y_label=None, bins='auto', color=UCD_COLORS['gold'],
         alpha=0.7, rwidth=0.85):
    assert isinstance(data, np.ndarray)
    n, bins, patches = plt.hist(x=data, label=label, bins=bins, color=color,
                                alpha=alpha, rwidth=rwidth)
    plt.grid(axis='y', alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


def scatter(x, y, title=None, x_label=None, y_label=None, alpha=0.75, save_as='plot.png', c=None):
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


def multi_hist(data, title, x_label, y_label, labels, bins='auto', colors=None,
               alpha=0.7, rwidth=0.85, save_as='plot.png'):
    save_as = os.path.join(PLOTS_DIR, save_as)
    assert isinstance(data, list)
    maxfreq = 0
    for i in range(len(data)):
        if colors is None:
            n, _, _ = plt.hist(data[i], bins=bins, alpha=alpha,
                               label=labels[i], rwidth=rwidth)
        else:
            assert len(colors) == len(data)
            n, _, _ = plt.hist(data[i], bins=bins, color=colors[i],
                               alpha=alpha, label=labels[i],
                               rwidth=rwidth)

    plt.grid(axis='y', alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.savefig(save_as, dpi=300)


def line_plot_df(data, x, y, title=None, xlabel=None, ylabel=None):
    ax = data.plot.line(x, y, colors=[UCD_COLORS['black'],
                                      UCD_COLORS['dark_gray'],
                                      UCD_COLORS['gold']])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
