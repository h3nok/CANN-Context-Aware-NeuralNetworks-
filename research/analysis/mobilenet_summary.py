from enum import Enum
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class MobileNetLoss(Enum):
	regularization = "regularization_loss_1"
	total = "total_loss_1"
	softmax = "losses/softmax_cross_entropy_loss/value"


class MobileNetV1Summary(object):
	event_file = ""
	tags = []

	def __init__(self, event_file_path):
		assert event_file_path is not None, "Must supply a valid events file\n"
		self.event_file = event_file_path

	def process_loss(self):
		event_acc = EventAccumulator(self.event_file)
		event_acc.Reload()
		self.tags = event_acc.Tags()
		import pprint
		pprint.pprint(event_acc.Tags())


if __name__ == "__main__":
	event_file = "E:\\Thesis\\CC_V2\\summaries\\curriculum\\mobilenet_v1\\cifar10\\baseline\\100000\\events.out.tfevents.1553615450.node18"
	event_summary = MobileNetV1Summary(event_file)

	event_summary.process_loss()
