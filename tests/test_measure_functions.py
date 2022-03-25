from unittest import TestCase
import numpy as np
from deepclo.core.measures.measure_functions import map_measure_function,\
    MeasureType, Measure


class TestMeasureFunctions(TestCase):
    def test_map_measure_fn(self):
        patch_1 = np.array([-1, 1, 1, 1])
        patch_2 = np.array([1, 1, 1, 1])
        patch_3 = np.array([1, 1, 0, 0])
        image_patches = [patch_1, patch_2, patch_3]
        m_func = map_measure_function(Measure.JE, MeasureType.DISTANCE)
        print(m_func(image_patches))
        m_func = map_measure_function(Measure.ENTROPY, MeasureType.STANDALONE)
        print(m_func(patch_1))


