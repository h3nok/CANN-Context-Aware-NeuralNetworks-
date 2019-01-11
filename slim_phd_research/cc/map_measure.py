from enum import Enum
from measures.distance import mi
from measures.distance import je
from measures.distance import kl
from measures.distance import ce
from measures.standalone import entropy
import functools
import tensorflow as tf


class Measure(Enum):
    MI = "Mutual Information"
    JE = "Joint Entropy"
    CE = "Conditional Entropy"
    E = "Standalone Entropy"


class MeasureType(Enum):
    Dist = "Distance measure between two patches"
    SA = "Standalone measure of a patch"


measure_map = {
    Measure.MI: mi.MutualInformation,
    Measure.JE: je.JointEntropy,
    Measure.CE: ce.ConditionalEntropy,
    Measure.E: entropy.Entropy
}


def map_measure_fn(m, measureType=MeasureType.Dist):
    if not isinstance(m, Measure):
        raise ValueError(
            "Supplied argument must be an instance of Measure Enum")

    if m not in measure_map:
        raise ValueError("Meaure function %s not found!" % m)

    assert isinstance(measureType, MeasureType), "Unknown measure type"

    func = measure_map[m]

    # Call functions by reflection
    @functools.wraps(func)
    def measure_fn(patches):
        if measureType == MeasureType.Dist:
            return func(patches[0], patches[1])
        else:
            return func(patches)

    return measure_fn


def get_measure_fn_test(patches):
    m_func = map_measure_fn(Measure.JE, MeasureType.Dist)

    return m_func(patches)


if __name__ == '__main__':
    patches = {}
    get_measure_fn_test(patches)
