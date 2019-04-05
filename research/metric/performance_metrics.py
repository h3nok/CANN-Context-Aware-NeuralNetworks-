from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import numpy

Measure = {
	'MI': "Mutual Information",
	'Cross Entropy': "Cross Entropy",
	'JE': "Joint Entropy",
	'CE': 'Conditional Entropy',
	'L1': 'L1-Norm',
	'L2': 'L2-Norm',
	'MN': 'Max-Norm',
	'SSIM': "Structural Similarity Index",
	'PSNR': "Peak-signal-to-noise-ratio",
	'MIN': "Normalized Mutual Information",
	'IV': "Information Variation",
	'KL': "Kullback-Liebler Divergence",
	'Baseline': "Baseline (No-Curriculum)"
}


class CsvData(object):
	source_file = None
	source_model = None
	data_frame = None
	columns = list()

	def __init__(self, source_file, source_model):
		self.source_file = source_file
		self.source_model = source_model

	def read(self):
		self.data_frame = read_csv(self.source_file).dropna()
		self.columns = list(self.data_frame)


if __name__ == '__main__':
	file = "E:\Thesis\OneDrive\PhD\Publications\Deep " \
	       "Learning\CC2\BMVC\Data\MobileNet\\run_baseline_100000-tag-total_loss_1.csv "
	model = 'MobileNet'

	csv = CsvData(file, model)
	csv.read()

	for key, value in Measure.items():
		plt.plot('Step', key, data=csv.data_frame.rolling(window=2).mean(), marker='o')
	plt.legend()
	plt.show()


