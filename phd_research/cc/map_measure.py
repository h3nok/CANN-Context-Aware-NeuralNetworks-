import functools
from enum import Enum

import tensorflow as tf

try:
    from cc.measures.distance import ce, je, kl, l1_norm, l2_norm, max_norm, mi, ssim, psnr
    from cc.measures.standalone import entropy
    from cc.cc_utils import ConfigureLogger
except (Exception, ImportError) as error:
    print(error)
    from measures.distance import ce, je, kl, l1_norm, l2_norm, max_norm, mi, ssim, psnr
    from measures.standalone import entropy
    from cc_utils import ConfigureLogger

_logger = ConfigureLogger(__file__, '.')


class Measure(Enum):
    MI = ('mi', "Mutual Information")
    JE = ('je', "Joint Entropy")
    CE = ('ce', "Conditional Entropy")
    ENTROPY = ('e',"Standalone Entropy")
    KL = ('kl',"Kullback-Leibler Divergence")
    L1 = ('l1',"L1-Norm")
    L2 = ('l2', "L2-Norm")
    SSIM = ('ssim',"Structural Similarity Index")
    PSNR = ('psnr',"Peak-Signal-to-Noise ratio")
    MAX_NORM = ('mn',"Max norm")


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

    _logger.info("Entering map_measure_fn, measure: {}".format(m.value))
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

    _logger.info("Successfully mapped measure function")

    return measure_fn


def get_measure_fn_test(patches):
    m_func = map_measure_fn(Measure.JE, MeasureType.Dist)
    return m_func(patches)


# if __name__ == '__main__':
#     patches = {}
#     get_measure_fn_test(patches)
