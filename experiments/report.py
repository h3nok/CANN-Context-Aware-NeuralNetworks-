import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from functools import reduce
import numpy as np


class Colors:

    @staticmethod
    def gold_1():
        return "#7B6F4B"

    @staticmethod
    def cu_gold():
        return "#CFB87C"

    @staticmethod
    def gold_plus_1():
        return "#D6CCA6"

    @staticmethod
    def white():
        return "#FFFFFF"

    @staticmethod
    def black():
        return "#000000"

    @staticmethod
    def black_plus_1():
        return "#595955"

    @staticmethod
    def black_plus_2():
        return "#7C7E7F"

    @staticmethod
    def black_plus_3():
        return "#9EA1A2"

    @staticmethod
    def cu_colors():
        return [Colors.cu_gold(), Colors.gold_1(),
                Colors.gold_plus_1(), Colors.black_plus_1(),
                Colors.black_plus_2(), Colors.black_plus_3()]


def read_results(results_path, model='b0'):
    files = os.listdir(results_path)
    optimizer = ''
    experiment = ''
    train_data, val_data = [], []

    for file in files:
        if not file.endswith('csv'):
            continue

        filename_split = file.split('_')
        if 'baseline' in filename_split:
            syllabus = 'baseline'
        else:
            syllabus = filename_split[2]

        optimizer = filename_split[0]
        experiment = filename_split[1]

        dataframe = pd.read_csv(os.path.join(results_path, file))
        dataframe = dataframe.rename(columns={'Value': syllabus})
        dataframe = dataframe.drop(columns=['Wall time'])
        if 'train' in file:
            train_data.append(dataframe)
        else:
            val_data.append(dataframe)

    assert len(train_data) == len(val_data)
    train_df = reduce(lambda x, y: pd.merge(x, y, on='Step'), train_data)
    train_df.to_csv(os.path.join('../experiments/results/', model + f'_{experiment}_train.csv'), index=False)
    val_df = reduce(lambda x, y: pd.merge(x, y, on='Step'), val_data)
    val_df.to_csv(os.path.join('../experiments/results/', model + f'_{experiment}_val.csv'), index=False)

    return train_df, val_df


class Report:
    def __init__(self, train_df, model, dataset, optimizer, val_df=None, split='Train'):
        self.dataframe = train_df.dropna()
        self.dataframe_val = val_df.dropna()
        self.columns = list(self.dataframe.columns)
        self.columns.remove('Step')
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.split = split
        self.line_styles = []
        self.max_values = {}

        sns.set_style('whitegrid')
        sns.set_palette("bright", 10, 1)

        self.line_plot(split='Train', save_fig=True)
        self.line_plot(split='Validation', save_fig=True)

        self.max_values = {k: v for k, v in sorted(self.max_values.items(), key=lambda item: item[1])}
        print(self.max_values)
        self.box_plot()

    def line_plot(self, save_fig=None, split='Train'):

        if split:
            self.split = split
        else:
            assert self.split

        assert split in ['Train', 'Validation']

        fig, ax = plt.subplots()
        line_styles = []
        n = self.dataframe_val.shape[0]

        if split == 'Train':
            for c in self.columns:
                if 'baseline' in c:
                    line_styles.append('--')
                else:
                    line_styles.append('-')

            self.dataframe.plot.line(x='Step',
                                     y=self.columns,
                                     grid=True,
                                     style=line_styles,
                                     lw=2, ax=ax)
        elif split == 'Validation':
            val_columns = list(self.dataframe_val.columns)
            val_columns.remove('Step')
            line_styles.clear()

            for c in val_columns:
                if 'baseline' in c:
                    line_styles.append('--')
                else:
                    line_styles.append('-')

                self.max_values[c.upper()] = self.dataframe_val[c][n-1]

            self.dataframe_val = self.dataframe_val.rolling(10).mean()
            self.dataframe_val.plot.line(x='Step',
                                         y=val_columns,
                                         grid=True,
                                         style=line_styles,
                                         lw=2, ax=ax)

        ax.set_title(f"{self.model} {self.split} Performance ({self.dataset}, {self.optimizer})")

        plt.ylabel("Accuracy")

        if save_fig:
            plt.savefig(os.path.join('../experiments/results/',
                                     f'{self.model}_{self.dataset}_{self.optimizer}_{self.split}.png'),
                        dpi=1000)

    def box_plot(self):
        fig, ax = plt.subplots()
        data = pd.DataFrame(self.max_values.items(), columns=['Syllabus', 'Acc'], index=None)
        sns.histplot(data=data, x='Acc', y='Syllabus', hue='Syllabus')
        print(np.argmax(dataset.y_test, axis=1))
        ax.get_legend().remove()
        plt.axhline(y='baseline'.upper(), lw=1, ls='--', color=Colors.gold_1())
        ax.set_title(f"{self.model}  Validation Accuracy Comparison ({self.dataset}, {self.optimizer})")
        plt.savefig(os.path.join('../experiments/results/',
                                 f'{self.model}_{self.dataset}_{self.optimizer}_{self.split}_val_hist.png'),
                    dpi=1000)
        data.to_csv(os.path.join('../experiments/results/',
                                 f'{self.model}_{self.dataset}_{self.optimizer}_{self.split}_va.csv'),
                    index=False)


if __name__ == "__main__":
    csv_dir = r'C:\deepclo\report\CIFAR10\Adam\b0\CLO'
    model = 'EfficientNet-B0'
    dataset = 'CIFAR10'
    optimizer = 'Adam'
    train, val = read_results(csv_dir)
    report = Report(train_df=train, model=model, val_df=val, dataset=dataset, optimizer=optimizer)
