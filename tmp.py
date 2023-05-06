import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc, rcParams

rc('font', weight='bold', )
rcParams["font.family"] = "Times New Roman"
plt.rcParams['grid.color'] = 'black'
plt.rcParams['axes.grid'] = True
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
def create_trend_plot(x_data, y1_data_list,
                      y2_data_list, labels,
                      title='Trend Plot', subtitle="", attack='X',
                      y1_label='Y1',
                      y2_label='Y2', output_file='trend_plot.png'):
    # Convert the input lists into a Pandas DataFrame
    df1 = pd.DataFrame({'X': x_data})
    for i, y_data in enumerate(y1_data_list):
        df1[f'Y1{i}'] = y_data

    df2 = pd.DataFrame({'X': x_data})
    for i, y_data in enumerate(y2_data_list):
        df2[f'Y2{i}'] = y_data

    # Calculate the standard deviation for each data set in y1_data_list
    y1_std_list = [np.std(y_data) for y_data in y1_data_list]

    # Customize the plot appearance
    # sns.set(style='whitegrid', font_scale=2)
    sns.set(style='whitegrid', font_scale=2, rc={'font.family': 'Arial'})
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Set bright color palette
    colors = sns.color_palette("bright", len(y1_data_list))

    # Create the trend plot for the first Y axis using Seaborn's lineplot function
    for i in range(len(y1_data_list)):
        linestyle = '--' if labels[i].lower() == 'undefended' else '-'
        sns.lineplot(x='X', y=f'Y1{i}', data=df1, lw=4, markersize=10, label=labels[i], ax=ax1,
                     linestyle=linestyle, marker='o', color=colors[i])

    # Customize the plot appearance for the first Y axis
    ax1.set_xlabel(attack, fontsize=22, fontweight='bold')
    ax1.set_ylabel(y1_label, fontsize=22, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=24, width=2, length=6)
    ax1.tick_params(axis='x', labelsize=24, width=2, length=6)

    # # Create a second Y axis on the right side of the plot
    # ax2 = ax1.twinx()
    #
    # # Set the range for the second Y axis
    # ax2.set_ylim(14, 100)
    #
    # # Customize the plot appearance for the second Y axis
    # ax2.set_ylabel(y2_label, fontsize=22, fontweight='bold')
    # ax2.tick_params(axis='y', labelsize=24, width=2, length=6)

    # Add a legend for both Y axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()


def plot_training_trend(csv_file, x_col, y_cols,
                        y_label='Accuracy(%)',
                        output_file='training_trend_plot.png',
                        x_label=None, highlight_col='Undefended',
                        dpi=300, figsize=(8, 4)):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Set bright color palette
    colors = sns.color_palette("bright", len(y_cols))

    # Set plot style and context for publication
    sns.set(style='whitegrid', font_scale=2, rc={'font.family': 'Arial'})
    # sns.set(style='white', font_scale=2, rc={'font.family': 'Times New Roman'})

    # Initialize the plot
    plt.figure(figsize=figsize, dpi=dpi)

    # Loop through the list of y-columns and create a lineplot for each
    for i, y_col in enumerate(y_cols):
        if y_col == highlight_col:
            sns.lineplot(data=data, x=x_col, y=y_col, label=y_col, linewidth=4, linestyle='--', color='red')
        else:
            sns.lineplot(data=data, x=x_col, y=y_col, label=y_col, linewidth=4, color=colors[i])

    # Customize the plot
    plt.xlabel(x_label if x_label else x_col, fontsize=22, fontweight='bold')
    plt.ylabel(y_label, fontsize=22, fontweight='bold')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    legend_properties = {'weight': 'bold'}
    plt.legend(fontsize=14, prop=legend_properties)

    # Customize x-axis and y-axis ticks
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=6)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

    # Save the plot as a high-resolution image file
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()


def moving_average(data, window=1):
    return np.convolve(data, np.ones(window), 'valid') / window


def plot_beta_vs_patch_size(networks, beta_patch_size, patch_size, x_label='Patch Size',
                            output_file='beta_plot.png', dpi=300, figsize=(8, 4), smoothing=False,
                            smoothing_window=2):
    data_patch_size = pd.DataFrame()

    for i, network in enumerate(networks):
        for j, size in enumerate(patch_size):
            data_patch_size = data_patch_size.append({'Network': network, 'β': beta_patch_size[i][j], x_label: size},
                                                     ignore_index=True)

    sns.set(style='whitegrid', font_scale=2, rc={'font.family': 'Arial'})
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    bright_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8']

    for i, network in enumerate(networks):
        subset = data_patch_size[data_patch_size['Network'] == network]
        if smoothing:
            y_data = moving_average(subset['β'].values, smoothing_window)
            x_data = subset[x_label][:len(y_data)]
        else:
            y_data = subset['β']
            x_data = subset[x_label]
        sns.lineplot(x=x_data, y=y_data, lw=4, label=network, ax=ax,
                     color=bright_colors[i], marker='o', markersize=10)

    ax.set_xlabel(x_label, fontsize=20, fontweight='bold')
    ax.set_ylabel('β', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=20, width=3, length=8)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.legend(fontsize=20)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_defense_accuracy(defense, acc_data, attacks, dpi=300, figsize=(8, 4), font_scale=2,
                          save_path="defense_vs_accuracy.png"):
    # Create a DataFrame
    data = pd.DataFrame(acc_data, columns=attacks,
                        index=defense).reset_index().melt(id_vars='index',
                                                          var_name='Attack', value_name='Accuracy')
    data.columns = ['Defense', 'Attack', 'Accuracy']

    # Set high-resolution plot settings
    sns.set(rc={'figure.figsize': figsize})
    sns.set(font_scale=font_scale)
    sns.set_style("whitegrid")

    # Create the plot
    ax = sns.lineplot(x='Attack', y='Accuracy', hue='Defense', data=data, marker='o', markersize=10)
    # ax.set_title('Attack vs Accuracy for Different Defenses')

    # Save the plot as a high-resolution image
    plt.savefig(save_path, dpi=dpi)

    # Show the plot
    plt.show()


def create_time(csv_file, output_file, dpi=300):
    df = pd.read_csv(csv_file)

    # Convert wall-time to hours
    df["Inception(Undefended)"] = (df["Inception(Undefended)"] - df["Inception(Undefended)"].iloc[0]) / 3600
    df["Inception(Defense=MI)"] = (df["Inception(Defense=MI)"] - df["Inception(Defense=MI)"].iloc[0]) / 3600

    sns.set(style='whitegrid', rc={'font.family': 'Arial'})
    plt.figure(figsize=(8, 4))

    sns.lineplot(x="Epoch", y="Inception(Undefended)",
                 data=df, linewidth=4,
                 label="Inception (Undefended)", color="yellow")
    sns.lineplot(x="Epoch", y="Inception(Defense=MI)",
                 data=df, linewidth=4,
                 label="Inception (Defense=MI)", color='red')

    plt.ylabel("Wall-time (hours)", fontsize=20, fontweight='bold')
    plt.xlabel("Epoch", fontsize=20, fontweight='bold')
    plt.legend(fontsize=15)
    # ax1 = plt.gca()
    # ax2 = ax1.twinx()
    # ax2.set_ylabel("Wall-time (hours)", fontsize=22, fontweight='bold')
    # ax2.set_yticks(ax1.get_yticks())
    # ax2.set_ylim(ax1.get_ylim())

    plt.savefig(output_file, dpi=dpi)
    plt.close()


def compute_robusntess_score(acc_clean: list, acc_attack: list):
    result = [[round(attack - clean, 3) for attack in attack_row] for clean, attack_row in zip(acc_clean, acc_attack)]
    assert (len(acc_clean) == len(result))
    return result


def compute_defense_success_rate(correct_predictions, total):
    return round((correct_predictions / total) * 100, 3)


def plot_gen_performance(output_dir, model, attack, Defense, clean_acc, attack_acc, N):
    robustness_scores = compute_robusntess_score(clean_acc, attack_acc)
    print(robustness_scores)

    ylabel = "\u03B4"
    subscript_string = 'clean'
    y_2_label = r'$ACC_{attack}$'

    os.makedirs(os.path.join(output_dir, model), exist_ok=True)
    output_file = os.path.join(output_dir, model, f'{model}_{subscript_string}_{attack}_trend_plot.png')

    create_trend_plot(N, robustness_scores, attack_acc, Defense, y1_label=ylabel,
                      attack=attack, y2_label=y_2_label,
                      output_file=output_file,
                      title=f'{model}')
    print(output_file)


# ------------------------------------------------------

Defense = ['Undefended', "KL", "MI", "MN", "H", "PSNR"]
acc_attack_b0_n_pixel = [
    [44.3, 40.7, 35.2, 23.1, 21.2, 19.3, 14.9, 14.1, 12.4],
    [82.6, 83.12, 79.4, 72.1, 70.3, 64.2, 65.8, 57.1, 52.23],
    [91.3, 86.2, 81.4, 79.3, 75.2, 70.2, 65.1, 63.2, 61.2],
    [81.6, 80.2, 75.3, 64.3, 60.2, 59.2, 55.3, 52.1, 49.6],
    [86.2, 85.12, 83.5, 82.1, 79.4, 76.2, 75.2, 70.5, 69.5],
    [89.3, 87.2, 85.23, 72.3, 70.5, 68.34, 75.3, 72.9, 70.1]
]
acc_attack_b0_patch = [
    [24.3, 20.7, 15.2, 13.1, 11.2],
    [82.6, 83.12, 69.4, 32.1, 20.3],
    [91.3, 86.2, 71.4, 69.3, 55.2],
    [71.6, 65.2, 55.3, 54.3, 40.2],
    [86.2, 82.12, 81.5, 79.1, 69.4],
    [79.3, 77.2, 75.23, 72.3, 60.5]
]
acc_clean_b0_n_pixel = [98, 97.9, 98.5, 98.2, 95.4, 98]
defense_success_rate = []
N = [1, 2, 4, 6, 8, 10, 12, 14, 16]
Patch_size = [5, 10, 15, 20, 25]

output_dir = r"C:\Users\Henok\OneDrive\Research\Thesis\Thesis\Publications\BMVC2023\figures"
model = "EfficientNet-B0"
csv_file_path = r"C:\github\clo\clozoo\CIFAR10\b0_cifar10_adam_poch_loss.csv"
output_file_name = 'b0_training_trend.png'
output_file_name = os.path.join(output_dir, model, output_file_name)
plot_training_trend(csv_file_path, x_col='Epoch', y_cols=['Undefended', 'MI', "PSNR", "KL", "Entropy"],
                    dpi=300, y_label="Loss", output_file=output_file_name, figsize=(8, 4))

csv_file_path = r"C:\github\clo\clozoo\CIFAR10\b0_cifar10_adam_validation-acc.csv"
output_file_name_success = os.path.join(output_dir, model, 'b0_training_success_rate.png')
plot_training_trend(csv_file_path, x_col='Epoch', y_cols=['Undefended', 'MI', "MN", "Entropy"],
                    dpi=300, output_file=output_file_name_success, figsize=(8, 4))

plot_gen_performance(output_dir=output_dir, model=model,
                     Defense=Defense, attack='N=',
                     clean_acc=acc_clean_b0_n_pixel,
                     attack_acc=acc_attack_b0_n_pixel, N=N)
plot_gen_performance(output_dir=output_dir, model=model,
                     Defense=Defense, attack='Patch Size (%)=',
                     clean_acc=acc_clean_b0_n_pixel,
                     attack_acc=acc_attack_b0_patch,
                     N=Patch_size)

networks = ['ResNet(MI)', 'VGG(H)', 'Inception(PSNR)', 'EfficientNet(MI)']
patch_size = [5, 10, 15, 20]
n = [1, 2, 4, 16]

beta_patch_size = [
    [71.3, 66.2, 61.4, 49.3],
    [71.6, 65.2, 55.3, 54.3],
    [76.2, 62.12, 51.5, 49.1],
    [79.3, 67.2, 65.23, 52.3]
]

beta_n = [
    [91.3, 86.2, 71.4, 69.3],
    [91.6, 85.2, 75.3, 64.3],
    [86.2, 82.12, 73.5, 64.1],
    [89.3, 87.2, 75.23, 72.3]
]

plot_beta_vs_patch_size(networks, beta_patch_size, patch_size, x_label='Patch Size (%)=',
                        output_file=os.path.join(output_dir, 'beta_patch_size.png'))
plot_beta_vs_patch_size(networks, beta_n, n, x_label="N=",
                        output_file=os.path.join(output_dir,
                                                 'beta_npixel_size.png'))
create_time(csv_file=r"C:\github\clo\clozoo\CIFAR10\walltime_imagenet_adam_train_acc.csv",
            output_file=os.path.join(output_dir, 'walltime_imagenet_adam_train_acc.png'))
