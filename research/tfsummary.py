import tensorflow as tf
from enum import Enum
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
import numpy as np


class InceptionV2Loss(Enum):
	regularization = "regularization_loss_1"
	total = "total_loss_1"
	softmax = "losses/softmax_cross_entropy_loss/value"


METRIC_FULL_NAME = {
	'ce': "Conditional Entropy",
	'mn': "Max Norm",
	'ssim': "Structural Similarity Index",
	'psnr': 'Peak-Signal-to-Noise-Ratio',
	'min': 'Normalized Mutual Information',
	'mi': 'Mutual Information',
	'e': "Entropy",
	'kl': "Kullback-Leibler Divergence",
	'l2': 'L2-Norm',
	'l1': 'L1-Norm',
	'je': 'Joint Entropy',
	'iv': 'Information Variation',
	'baseline': 'Baseline (no-curriculum)',
	'base': 'Baseline (no-curriculum)',
	'cross_entropy': 'Cross Entropy',
	'cep': 'Cross Entropy PMF'
}


class Summary(object):
	event_file = ""
	tags = []

	def __init__(self, event_file_path, training=True):
		assert event_file_path is not None, "Must supply a valid events file\n"
		self.event_file = event_file_path

	def process(self):
		event_acc = EventAccumulator(self.event_file)
		event_acc.Reload()
		self.tags = event_acc.Tags()

		total_loss_summary = event_acc.Scalars(InceptionV2Loss.total.value)
		regularization_loss_summary = event_acc.Scalars(InceptionV2Loss.regularization.value)
		softmax_loss_summary = event_acc.Scalars(InceptionV2Loss.softmax.value)

		wall_time = []
		steps = []
		total_loss = []
		reg_loss = []
		softmax_loss = []

		for item in total_loss_summary:
			wall_time.append(item[0])
			steps.append(item[1])
			total_loss.append(item[2])

		for item in regularization_loss_summary:
			reg_loss.append(item[2])

		for item in softmax_loss_summary:
			softmax_loss.append(item[2])

		return steps, total_loss, reg_loss, softmax_loss


def plot_multiple(data, plot='Softmax loss', network='Inception V2', outdir=".", format="png"):
	for run in data:
		plt.plot(run['Steps'], run[plot], marker='o',
									label=run['Metric'], c=np.random.rand(3))

	plt.xlabel("Training Steps")
	plt.ylabel("Loss")
	plt.title("{} {}".format(network, plot))
	plt.legend(loc='upper right', frameon=True, fontsize='small')
	plt.savefig(os.path.join(outdir, '{}_{}.{}'.format(plot, network, format)), format=format, dpi=1000)
	plt.show()
	plt.close()


if __name__ == '__main__':
	summaries_dir = "E:\\Thesis\\CC_V2\\summaries\\curriculum"
	output_dir = "E:\\Thesis\\CC_V2\\summaries\\plots"
	iterations = 10000
	training_data = []
	data = {}
	for measure in os.listdir(summaries_dir):
		metric = measure
		event_file_dir = os.path.join(summaries_dir, metric, str(iterations))
		files = os.listdir(event_file_dir)
		event_file = None
		for file in files:
			if file.endswith('.node18'):
				event_file = file
		if not event_file: continue
		log_file = os.path.join(event_file_dir, event_file)
		s = Summary(log_file)
		steps, loss, reg_loss, softmax_loss = s.process()
		data = {
			"Steps": steps,
			"Total loss": loss,
			"Metric": METRIC_FULL_NAME[metric],
			"Reg loss": reg_loss,
			"Softmax loss": softmax_loss
		}
		training_data.append(data)

	plot_multiple(training_data, outdir=output_dir)
	plot_multiple(training_data, plot="Total loss", outdir=output_dir)
	plot_multiple(training_data, plot="Reg loss", outdir=output_dir)
