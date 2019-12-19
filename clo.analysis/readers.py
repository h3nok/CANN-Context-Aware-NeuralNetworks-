import pandas as pd
import matplotlib.pyplot as plt
from plots import line_plot_df
import os


class ExcelImporter(object):
    _file_path = None
    _pd_dataframe = None
    _head = None
    columns = None

    def __init__(self, file_path):
        self._file_path = file_path

    def import_pd(self):
        self._pd_dataframe = pd.read_excel(self._file_path)
        self._head = self._pd_dataframe.head()
        self.columns = list(self._pd_dataframe.columns)
        assert isinstance(self._pd_dataframe, pd.core.frame.DataFrame)

    @property
    def pd_dataframe(self):
        return self._pd_dataframe


class CSV:
    _filepath = None
    _df = None

    def __init__(self, file):
        self._filepath = file

    def to_df(self):
        self._df = pd.read_csv(self._filepath)
        return self._df

    @property
    def columns(self):
        return self._df.colums


if __name__ == '__main__':
    savedir = r"E:\Thesis\OneDrive\Research\Publications\Deep Learning\2020\Dataset"
    excel = CSV(r"E:\Thesis\OneDrive\Research\Publications\Deep "
                r"Learning\2020\Training and Generalization\MI_SSIM_Training.csv")
    excel.to_df()
    line_plot_df(excel.to_df(), x='Step', y=['SSIM', 'MI', 'Baseline'], xlabel='Step', ylabel='Loss',
                 title='Training performance of MI and SSIM syllabus on CIFAR10 dataset',
                 saveas=os.path.join(savedir, 'training_loss_inception_smoothed.png'), smoothing='rolling')
    line_plot_df(excel.to_df(), x='Step', y=['SSIM', 'MI', 'Baseline'], xlabel='Step', ylabel='Loss',
                 title='Training performance of MI and SSIM syllabus on CIFAR10 dataset',
                 saveas=os.path.join(savedir, 'training_loss_inception.png'), smoothing=None)
    line_plot_df(excel.to_df(), x='Step', y=['SSIM', 'MI', 'Baseline'], xlabel='Step', ylabel='Loss',
                 title='Training performance of MI and SSIM syllabus on CIFAR10 dataset',
                 saveas=os.path.join(savedir, 'training_acc_inception.png'), smoothing='expanding')
    plt.show()
