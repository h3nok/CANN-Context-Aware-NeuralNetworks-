from enum import Enum
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class InceptionV2Loss(Enum):
	regularization = "regularization_loss_1"
	total = "total_loss_1"
	softmax = "losses/softmax_cross_entropy_loss/value"
	mixed_4c_sparsity = 'sparsity/Mixed_4c'
	mixed_3b_sparsity = 'sparsity/Mixed_3b'
	mixed_5b_sparsity = 'sparsity/Mixed_5b'



class InceptionV2Summary(object):
	event_file = ""
	tags = []

	def __init__(self, event_file_path):
		assert event_file_path is not None, "Must supply a valid events file\n"
		self.event_file = event_file_path

	def process_sparsity(self):
		event_acc = EventAccumulator(self.event_file)
		event_acc.Reload()
		self.tags = event_acc.Tags()

		mixed_4c_summary = event_acc.Scalars(InceptionV2Loss.mixed_4c_sparsity.value)
		mixed_3b_summary = event_acc.Scalars(InceptionV2Loss.mixed_3b_sparsity.value)
		mixed_5b_summary = event_acc.Scalars(InceptionV2Loss.mixed_5b_sparsity.value)

		steps = []
		mixed_4c = []
		mixed_3b = []
		mixed_5b = []
		walltime = []

		for item in mixed_4c_summary:
			walltime.append(item[0])
			steps.append(item[1])
			mixed_4c.append(item[2])
		for item in mixed_3b_summary:
			mixed_3b.append(item[2])
		for item in mixed_5b_summary:
			mixed_5b.append(item[2])

		return steps, mixed_4c, mixed_3b, mixed_5b, walltime

	def process_loss(self):
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