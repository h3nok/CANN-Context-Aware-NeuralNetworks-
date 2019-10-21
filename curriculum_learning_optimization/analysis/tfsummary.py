import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from analysis.google_net_summary import InceptionV2Summary

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


def _plot_multiple(data, plot='Softmax loss', network='Inception V2', outdir=".", format="png"):
	for run in data:
		x = np.asarray(run['Steps'])
		xnew = np.linspace(x.min(), x.max(), 300)
		y = np.asarray(run[plot])
		spl = make_interp_spline(x, y, k=1)
		ynew = spl(xnew)
		plt.plot(x, y, label=run['Metric'], marker='o', c=np.random.rand(3))

	plt.xlabel("Training Steps")
	plt.ylabel("Loss")
	plt.title("{} {}".format(network, plot))
	plt.legend(loc='upper right', frameon=False, fontsize='small')
	plt.savefig(os.path.join(outdir, '{}_{}.{}'.format(plot, network, format)), format=format, dpi=1000)
	plt.show()
	plt.close()


def plot_losses(data, output_dir):
	_plot_multiple(data, outdir=output_dir)
	_plot_multiple(data, plot="Total loss", outdir=output_dir)
	_plot_multiple(data, plot="Regularization loss", outdir=output_dir)


def moving_average(data, window_size):
	window = np.ones(int(window_size)) / float(window_size)
	return np.convolve(data, window, 'same')


def write_csv(data, filename):
	assert isinstance(data, dict), "Data must be dict type"

	with open(filename, 'a+') as f:
		w = csv.writer(f)
		w.writerow(data.keys())
		w.writerows(zip(*data.values()))


def run(summaries_dir="E:\\Thesis\\CC_V2\\summaries\\curriculum\\", model='mobilenet_v1', dataset='cifar10',
		iterations=100000, output_dir="E:\\Thesis\\CC_V2\\summaries\\plots"):
	"""
	:param summaries_dir:
	:param model:
	:param dataset:
	:param iterations:
	:param output_dir:
	:return:
	"""
	training_data = []
	data = {}
	summaries_dir = os.path.join(summaries_dir, model, dataset)
	for measure in os.listdir(summaries_dir):
		metric = measure
		event_file_dir = os.path.join(summaries_dir, metric, str(iterations))
		if not os.path.exists(event_file_dir): continue
		files = os.listdir(event_file_dir)
		event_file = None
		for file in files:
			if file.endswith('.node18'):
				event_file = file
		if not event_file: continue
		log_file = os.path.join(event_file_dir, event_file)
		s = InceptionV2Summary(log_file)
		steps, loss, reg_loss, softmax_loss = s.process_loss()
		data = {
			"Metric": METRIC_FULL_NAME[metric],
			"Steps": steps,
			"Total loss": loss,
			"Regularization loss": reg_loss,
			"Softmax loss": softmax_loss
		}
		training_data.append(data)

	plot_losses(training_data, output_dir=output_dir)


if __name__ == "__main__":
	run()
