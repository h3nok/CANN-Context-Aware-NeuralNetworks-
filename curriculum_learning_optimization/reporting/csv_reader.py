from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd


class Columns(Enum):
    wall_time = 'Wall time'
    baseline = 'Baseline'
    cross_entropy = 'Cross Entropy'
    mutual_information = 'MI'
    conditional_entropy = 'CE'
    joint_entropy = 'JE'
    l1_norm = 'L1'
    l2_norm = 'L2'
    max_norm = 'MN'
    ssim = 'SSIM'
    psnr = 'PSNR'
    mutual_information_normalized = 'MIN'
    kl = 'KL'
    information_variation = 'IV'
    step = 'Step'


class CSVDataFile(object):
    _file_path = None
    _pd_dataframe = None
    _head = None
    columns = None

    def __init__(self, file_path):
        self._file_path = file_path

    def read(self):
        self._pd_dataframe = pd.read_csv(self._file_path)
        self._head = self._pd_dataframe.head()
        self.columns = list(self._pd_dataframe.columns)
        assert isinstance(self._pd_dataframe, pd.core.frame.DataFrame)

    @property
    def pd_dataframe(self):
        return self._pd_dataframe


file = "E:\\Thesis\\OneDrive\Research\Publications\Deep Learning\CC2\BMVC\Data\mobilenet_v1_loss.csv"
csv_data = CSVDataFile(file)
csv_data.read()

df = csv_data.pd_dataframe

data = {
    Columns.wall_time: df[Columns.wall_time.value],
    Columns.step: df[Columns.step.value],
    Columns.baseline: df[Columns.baseline.value],
    Columns.cross_entropy: df[Columns.cross_entropy.value],
    Columns.joint_entropy: df[Columns.joint_entropy.value],
    Columns.mutual_information: df[Columns.mutual_information.value],
    Columns.conditional_entropy: df[Columns.conditional_entropy.value],
    Columns.l1_norm: df[Columns.l1_norm.value],
    Columns.l2_norm: df[Columns.l2_norm.value],
    Columns.max_norm: df[Columns.max_norm.value],
    Columns.ssim: df[Columns.ssim.value],
    Columns.psnr: df[Columns.psnr.value],
    Columns.mutual_information_normalized: df[Columns.mutual_information_normalized.value],
    Columns.kl: df[Columns.kl.value],
    Columns.information_variation: df[Columns.information_variation.value]
}

plt.plot(Columns.step.value, Columns.baseline.value, data=df, color='olive')
plt.plot(Columns.step.value, Columns.cross_entropy.value, data=df, color='red')
plt.plot(Columns.step.value, Columns.mutual_information.value, data=df, color='green')
plt.plot(Columns.step.value, Columns.mutual_information_normalized.value, data=df, color='blue')
plt.legend()
plt.show()
