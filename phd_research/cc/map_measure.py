import functools
from enum import Enum

import tensorflow as tf

from measures.distance import ce, je, kl, l1_norm, l2_norm, max_norm, mi, ssim, psnr
from measures.standalone import entropy


class Measure(Enum):
    MI = "Mutual Information"
    JE = "Joint Entropy"
    CE = "Conditional Entropy"
    ENTROPY = "Standalone Entropy"
    KL = "Kullback-Leibler Divergence"
    L1 = "L1-Norm"
    L2 = "L2-Norm"
    SSIM = "Structural Similarity Index"
    PSNR = "Peak-Signal-to-Noise ratio"
    MAX_NORM = "Max norm"


class Ordering(Enum):
    Ascending = "Sort patches in ascending rank order"
    Descending = "Sort patches in descending rank order"


class MeasureType(Enum):
    Dist = "Distance measure between two patches"
    STA = "Standalone measure of a patch"


MEASURE_MAP = {
    Measure.MI: mi.MutualInformation,
    Measure.JE: je.JointEntropy,
    Measure.CE: ce.ConditionalEntropy,
    Measure.ENTROPY: entropy.Entropy,
    Measure.KL: kl.KullbackLeiblerDivergence,
    Measure.L1: l1_norm.L1Norm,
    Measure.L2: l2_norm.L2Norm,
    Measure.MAX_NORM: max_norm.MaxNorm,
    Measure.PSNR: psnr.PSNR,
    Measure.SSIM: ssim.SSIM
}


def map_measure_fn(m, measureType=MeasureType.Dist):
    """[summary]

    Arguments:
        m {Measure} -- measure to use

    Keyword Arguments:
        measureType {MeasureType} -- measure type (default: {MeasureType.Dist})
    """

    if not isinstance(m, Measure):
        raise ValueError(
            "Supplied argument must be an instance of Measure Enum")
    if not isinstance(measureType, MeasureType):
        raise ValueError(
            "Supplied argument must be an instance of MeasureType Enum")

    if m not in MEASURE_MAP:
        raise ValueError("Meaure function %s not found!" % m)

    assert isinstance(measureType, MeasureType), "Unknown measure type"

    func = MEASURE_MAP[m]

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
