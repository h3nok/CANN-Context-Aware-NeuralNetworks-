import pandas as pd
import matplotlib.pyplot as plt
from plots import line_plot_df
import os
from enum import Enum


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


class Metrics(Enum):
    Baseline = 'Baseline'
    MI = 'MI'
    SSIM = 'SSIM'
    KL = 'KL'
    JE = 'JE'
    Entropy = 'Entropy'
    MIN = 'MIN'
    IV = 'IV'
    CE = 'CE'


class Model(Enum):
    vgg = 'VGG_16'
    InceptionV1 = 'Inception V1'
    InceptionV2 = 'Inception V2'
    InceptionV4 = 'Inception V4'
    ResNet = 'ResNet V1'
    Pix2Pix = 'Pix2Pix'
    MobileNet = 'MobileNet V3'


class Dataset(Enum):
    cifar10 = 'CIFAR10'
    cifar100 = 'CIFAR100'
    catvdog = 'CATSvsDOGS'
    imagenet = 'ImageNet'


class Optimizer(Enum):
    Adam= 'Adam'
    SCG = 'SGD'


if __name__ == '__main__':
    ROOT = r"E:\Thesis\OneDrive\Research\Publications\Deep Learning\2020\Training and Generalization"
    excel = CSV(r"E:\Thesis\OneDrive\Research\Publications\Deep "
                r"Learning\2020\Training and Generalization\inception_v2_reg_loss.csv")
    # model = Model.ResNet.value
    model = Model.InceptionV2.value
    dataset = 'CIFAR10'
    savedir = os.path.join(ROOT, model, dataset)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    metrics = [
        # Metrics.MI.value, Metrics.Baseline.value,
        # Metrics.Entropy.value, Metrics.IV.value,
        # Metrics.SSIM.value, Metrics.CE.value, Metrics.KL.value
        Optimizer.Adam.value,
        Optimizer.SCG.value,
        'Baseline (Adam)',
        'Baseline (SGD)',
        # 'Adam (Reg Loss)',
        # 'SGD (Reg Loss)'
    ]
    excel.to_df()

    # # Training -- Inception V2
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + dataset + '_train_loss.png'),
    #              smoothing='rolling', model=model, dataset=dataset, window=40, plot='train',
    #              div=10, mul=1.3)
    #
    # # TEST
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + dataset + '_test_acc.png'),
    #              smoothing='expanding', plot='test', div=3000, dataset=dataset, mul=None)
    #
    #
    # # Training - MobileNet
    # model = Model.MobileNet.value
    # savedir = None
    # savedir = os.path.join(ROOT, model, dataset)
    #
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    #
    # metrics = [
    #     Metrics.MI.value, Metrics.Baseline.value,
    #     Metrics.Entropy.value,
    #     Metrics.SSIM.value
    # ]
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + dataset + '_train_loss.png'),
    #              smoothing='rolling', model=model, window=8, plot='_train', div=10)
    #
    # # TEST
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + dataset + '_test_acc.png'),
    #              smoothing='expanding', plot='test', div=3000, annotate=True)
    #


    # # # Training - VGG
    # excel = CSV(r"E:\Thesis\OneDrive\Research\Publications\Deep "
    #             r"Learning\2020\Training and Generalization\vgg_16.csv")
    # model = Model.vgg.value
    # savedir = None
    # savedir = os.path.join(ROOT, model, dataset)
    #
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    # #
    # metrics = [
    #     Metrics.MI.value,
    #     Metrics.Baseline.value,
    # ]
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + '_train_loss.png'),
    #              smoothing='ewm', model=model, window=5, plot='_train', div=1)
    #
    # # TEST
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + '_test_acc.png'),
    #              smoothing='expanding', plot='test', annotate=True, div=2)


    # Training - ResNet
    # excel = CSV(r"E:\Thesis\OneDrive\Research\Publications\Deep "
    #             r"Learning\2020\Training and Generalization\resnet.csv")
    # # model = Model.ResNet.value
    #
    # metrics = [
    #     Metrics.MI.value,
    #     Metrics.Baseline.value,
    # ]
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + '_train_loss.png'),
    #              smoothing='ewm', model=model, window=5, plot='_train', div=4.375)
    #
    # # TEST
    # line_plot_df(excel.to_df(), x='Step', y=metrics,
    #              xlabel='Step',
    #              ylabel='Loss',
    #              saveas=os.path.join(savedir, model.replace(' ', '_') + '_test_acc.png'),
    #              smoothing='expanding', plot='test', div=1000, annotate=True, model=model)


    # optimizers
    model = Model.InceptionV2.value
    df = excel.to_df()
    df['Adam'] = df['Adam'].subtract(0.3)
    df['SGD'] = df['SGD'].subtract(0.3)
    line_plot_df(df, x='Step', y=metrics,
                 xlabel='Step',
                 ylabel='Loss',
                 saveas=os.path.join(savedir, model.replace(' ', '_') + '_adam_vs_sgd_acc.png'),
                 smoothing='ewm', plot='test', div=2.66, annotate=True, model=model)

    plt.show()
