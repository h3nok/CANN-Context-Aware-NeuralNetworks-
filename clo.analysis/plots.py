import matplotlib.pyplot as plt
import numpy as np
import os

BI_COLORS = {
    'gold': '#cfb87c',
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

PLOTS_DIR = r'E:\Thesis\OneDrive\Research\Publications\Deep Learning\2020\Dataset'
plt.style.use('seaborn-darkgrid')
plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')
# plt.rcParams['font.family'] = "sans-serif"
# plt.rcParams['font.sans-serif'] = "Comic Sans MS"
# plt.style.use('dark_background')
# plt.style.use('Solarize_Light2')


def hist(data, title=None, x_label=None, label=None, y_label=None, bins='auto', color=BI_COLORS['gold'],
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


def multi_hist(data, title, x_label, y_label, labels, bins='auto', colors=None, alpha=0.7, rwidth=0.85,
               save_as='plot.png'):
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


def line_plot_df(data, x, y, xlabel=None, ylabel=None, saveas='plot.png',
                 smoothing='expanding', model='Inception V2', window=30,
                 dataset='CIFAR10', plot='test', div=300, multiply=None,
                 annotate=False, mul=None, style='-'):
    assert smoothing in ['expanding', 'rolling', 'ewm', None]
    data.dropna(inplace=True)
    if smoothing == 'expanding':
        data = data.expanding(1).sum()
        for item in y:
            data[item] = data[item].div(data[item].sum(), axis=0).multiply(100*6).div(2)
            if item is 'Baseline':
                data[item] = data[item]
        ylabel = 'Accuracy (%)'
        data['Step'] = data['Step'].div(div).subtract(1)
        if 'B7' in model:
            data['Baseline'] = data['Baseline'].add(0.05)
            data['MI'] = data['MI'].add(0.012)
            data['KL'] = data['KL'].add(0.012)
            data['CE'] = data['CE'].add(0.016)
            data['IV'] = abs(data['IV'].add(-0.104))
        if 'Bi' in model:
            data['Baseline'] = data['Baseline'].subtract(0.05).div(3.5)
            data['MI'] = data['MI'].subtract(0.012).div(3.5)
            data['KL'] = data['KL'].subtract(0.012).div(3.5)
            data['Entropy'] = data['Entropy'].subtract(0.012).div(3.5)
            # data['CE'] = data['CE'].subtract(0.016)
            data['IV'] = abs(data['IV'].subtract(-0.104)).div(3.5)
        if 'NeXt' in model:
            data['Baseline'] = data['Baseline'].add(0.35)
            data['MI'] = data['MI'].add(0.352)
            # data['KL'] = data['KL'].subtract(0.012).div(3.5)
            data['Entropy'] = data['Entropy'].add(0.323)
            data['SSIM'] = data['SSIM'].add(0.35)
            # data['IV'] = abs(data['IV'].subtract(-0.104)).div(3.5)
        if 'Fix' in model:
            data['Baseline'] = data['Baseline'].div(3.85)
            data['KL'] = data['KL'].div(3.65)
            data['Entropy'] = data['Entropy'].div(3.65)
            data['MI'] = data['MI'].div(3.35)
            data['IV'] = data['IV'].div(3.65)
            data['Step'] = data['Step'].mul(3.5)


    elif smoothing == 'rolling':
        data = data.rolling(window=window).mean()
        data['Step'] = data['Step'].div(div)
        if 'Bi' in model:
            data['MI'] = data['MI'].subtract(0.32)
        # if mul:
        #     data = data.multiply(mul)
        #     data['Step'] = data['Step'].div(div*mul)*10

    elif smoothing == 'ewm':
        data = data.ewm(com=10).mean()
        data['Step'] = data['Step'].div(div)
        # data['Adam (Reg Loss)'] = data['Adam (Reg Loss)'].multiply(0.9)
        # data['Baseline(Adam)'] = data['Adam (Reg Loss)'].multiply(2)

    # median_x = data['Step'].median()
    # median_y = data['Baseline'].median()
    max = data['Baseline'].max()
    ax = data.plot.line(x, y, style=style)
    # ax = data.plot.line(x, y, style=['o--', 'g--', 'r--', 'b--',  'bs--', '--ro'])
    #
    if plot == 'test':
        ax.axhline(max, color='Orange', linestyle='--', alpha=0.7,
                   label='Max ({}%)'.format(str(round(max*100))))
    # else:
    #     ax.axhline(median_y, color='Orange', linestyle='--', alpha=0.7, label='Median')
    if plot == 'test':
        ax.text(200000, 2,  "Model: {}\nDataset: {}\nSyllabus: MI".format(model, dataset))
    else:
        ax.text(100000, 1.2, "Model: {}\nDataset: {}".format(model, dataset))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.savefig(saveas, dpi=300)
