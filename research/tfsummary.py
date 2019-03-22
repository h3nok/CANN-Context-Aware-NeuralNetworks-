import tensorflow as tf
from enum import Enum
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

class InceptionV2Loss(Enum):
	regularization = "regularization_loss_1"
	total = "total_loss_1"
	softmax = "losses/softmax_cross_entropy_loss/value"


class Summary(object):
	event_file = ""
	tags = []
	wall_time = []
	steps = []
	total_loss = []
	reg_loss = []
	softmax_loss = []

	def __init__(self, event_file_path, network="", measure="",
					dataset="", iterations="", training=True):
		assert event_file_path is not None, "Must supply a valid events file\n"
		self.event_file = event_file_path

	def process(self):
		event_acc = EventAccumulator(self.event_file)
		event_acc.Reload()
		self.tags = event_acc.Tags()

		total_loss_summary = event_acc.Scalars(InceptionV2Loss.total.value)
		regularization_loss_summary = event_acc.Scalars(InceptionV2Loss.regularization.value)
		softmax_loss_summary = event_acc.Scalars(InceptionV2Loss.softmax.value)

		for item in total_loss_summary:
			self.wall_time.append(item[0])
			self.steps.append(item[1])
			self.total_loss.append(item[2])

		for item in regularization_loss_summary:
			self.reg_loss.append(item[2])

		for item in softmax_loss_summary:
			self.softmax_loss.append(item[2])

	def plot_all_losses(self):
		plt.plot(self.steps, self.softmax_loss, label="Cross-Entropy Loss")
		plt.plot(self.steps, self.reg_loss, label="Regularization Loss")
		plt.plot(self.steps, self.total_loss, label="Total Loss")

		plt.xlabel("Training Steps")
		plt.ylabel("Loss")
		plt.title("Training Loss")
		plt.legend(loc='upper right', frameon=True)
		plt.show()

	def plot_total_loss(self, metric):
		plt.plot(self.steps, self.total_loss, label="Total Loss")

		plt.xlabel("Training Steps")
		plt.ylabel("Loss")
		plt.title("Training Loss, Metric: {}".format(metric))
		plt.legend(loc='upper right', frameon=True)
		plt.show()
		plt.close()
		self.total_loss.clear()
		self.steps.clear()


if __name__ == '__main__':
	summaries_dir = "E:\\Thesis\\CC_V2\\summaries"
	iterations = 10000
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
		s.process()
		s.plot_total_loss(metric)


