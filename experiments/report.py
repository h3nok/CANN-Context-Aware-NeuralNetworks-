import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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


class Report:
    def __init__(self, csv
                 ):
        sns.set_style('darkgrid')

        assert os.path.exists(csv)
        self.dataframe = pd.read_csv(csv)
        self.columns = ['CLO', 'Baseline', 'POR']
        self.columns_loss = ['CLO-Loss', 'Baseline-Loss', 'POR-Loss']

    def line_plot(self):
        self.dataframe.plot.line(x='Step', y=self.columns, grid=True).set_title('Entropy')

    def box_plot(self):
        fig, ax = plt.subplots()
        self.dataframe.boxplot(column=self.columns_loss, ax=ax)
        plt.ylabel('Loss')
        labels = [item.get_text().replace('-Loss', '') for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels)
        plt.text(2.5, 2.5, 'Model: B0\nDataset: CIFAR10', bbox=dict(facecolor=Colors.gold_1(), alpha=0.1))


if __name__ == "__main__":
    dir_path = r"C:\deepclo\report\Entropy_B0_CIFAR10__acc.csv"
    report = Report(dir_path)
    report.line_plot()
    # report.box_plot()
    plt.show()
