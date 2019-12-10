import pandas as pd
import matplotlib.pyplot as plt
from plots import line_plot_df


class Excel:
    _filepath = None
    _df = None

    def __init__(self, file):
        self._filepath = file

    def to_df(self):
        self._df = pd.read_excel(self._filepath, index_col=0)
        print(self._df)


class CSV:
    _filepath = None
    _df = None

    def __init__(self, file):
        self._filepath = file

    def to_df(self):
        self._df = pd.read_csv(self._filepath)
        print(self._df.columns)
        return self._df

    @property
    def columns(self):
        return self._df.colums


if __name__ == '__main__':
    excel = CSV(r"E:\Thesis\OneDrive\Research\Publications\Deep "
                r"Learning\2020\Training and Generalization\MI_SSIM_Training.csv")
    excel.to_df()
    line_plot_df(excel.to_df(), x='Step', y=['SSIM', 'MI', 'Baseline'], xlabel='Step', ylabel='Loss',
                 title='Training performance of MI and SSIM syllabus on CIFAR10 dataset')
    plt.show()