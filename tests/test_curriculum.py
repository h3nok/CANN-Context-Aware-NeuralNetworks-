from unittest import TestCase

import matplotlib.pyplot as plt

from deepclo.algorithms.curriculum import Curriculum
from deepclo.core.measures.measure_functions import Measure
from deepclo.pipe.dataset import ImageDataProvider
from deepclo.utils import show_images


class TestCurriculum(TestCase):
    dataset_name = 'cifar10'
    dataset = ImageDataProvider(dataset_name= dataset_name)

    # sample = dataset.x_train[0]
    label = dataset.y_train[0]
    batch_size = 8
    train_batch = dataset.x_train[:batch_size]
    batch_labels = dataset.y_train[:batch_size]
    curriculum = Curriculum(train_batch, batch_labels)

    entropy_syllabus = curriculum.syllabus(measure=Measure.ENTROPY, reference_imag_index=None)
    # curriculum.print_syllabus()
    batch_syllabus, ranks = curriculum.syllabus(measure=Measure.JE, reference_imag_index=None)
    # curriculum.print_syllabus()
    plt.show()
