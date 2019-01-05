from enum import Enum
from measures.distance import mi
from measures.distance import je
from measures.distance import kl
import functools
import tensorflow as tf


class Measure(Enum):
    MI = "Mutual Information"
    JE = "Joint Entropy"
    CE = "Conditional Entropy"


measure_map = {
    Measure.MI: mi,
    Measure.JE: je.je,
    Measure.CE: kl
}


def get_measure_fn(m):
    if not isinstance(m, Measure):
        raise ValueError(
            "Supplied argument must be an instance of Measure Enum")

    if m not in measure_map:
        raise ValueError("Meaure function %s not found!" % m)

    func = measure_map[m]

    # Call functions by reflection
    @functools.wraps(func)
    def measure_fn(patches, ordering):
        return func(patches, ordering)

    return measure_fn


def get_measure_fn_test():
    m_func = get_measure_fn(Measure.JE)
    patches = {}
    ordering = 0

    m_func(patches, ordering)


if __name__ == '__main__':
    get_measure_fn_test()
