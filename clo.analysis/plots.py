import matplotlib.pyplot as plt
import numpy as np

UCD_COLORS = {
    'gold': '#cfb87c',
    'black': '#000000',
    'dark_gray': '#565A5C',
    'light_gray': '#A2A4A3'
}


def hist(data, title=None, x_label=None, y_label=None, bins='auto', color=UCD_COLORS['gold'], alpha=0.7, rwidth=0.85):
    assert isinstance(data, np.ndarray)
    n, bins, patches = plt.hist(x=data, bins=bins, color=color,
                                alpha=alpha, rwidth=rwidth)
    plt.grid(axis='y', alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


def scatter(x, y, title=None, x_label=None, y_label=None, c=UCD_COLORS['dark_gray'], alpha=0.75):
    area = y**2
    plt.scatter(x, y,  s=area, c=np.random.rand(len(x)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)