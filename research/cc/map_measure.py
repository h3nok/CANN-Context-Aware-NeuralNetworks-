import functools
from enum import Enum

try:
	from cc.measures import l1_norm, l2_norm,max_norm, mi, ssim, psnr, entropy
	from cc.utils import ConfigureLogger
except (Exception, ImportError) as error:
    from measures import l1_norm, l2_norm, max_norm, mi,ssim, psnr, entropy
    from utils import ConfigureLogger

_logger = ConfigureLogger(__file__, '.')


class Measure(Enum):
    MI = ('mi', "Mutual Information")
    JE = ('je', "Joint Entropy")
    CE = ('ce', "Conditional Entropy")
    RE = ('re', "Relative Entropy")
    ENTROPY = ('e', "Entropy")
    KL = ('kl', "Kullback-Leibler Divergence")
    L1 = ('l1', "L1-Norm")
    L2 = ('l2', "L2-Norm")
    SSIM = ('ssim', "Structural Similarity Index")
    PSNR = ('psnr', "Peak-Signal-to-Noise ratio")
    MAX_NORM = ('mn', "Max norm")
    MI_NORMALIZED = ('min', "Normalized mutual information ")
    EI = ("ei", "Enigmatic information")
    LI = ("li", "Lautum information")
    MULTI_I = ("mui", "Multi-information")
    BI = ("bi", "Binding information")
    COI = ("coi", "Co-Information")
    CROSS_ENTROPY = ("cross_entropy", "Cross entropy")
    CROSS_ENTROPY_PMF = ("cep", "Cross entropy pmf")
    ELI = ("eli", "Exogenous local information")
    II = ("ii", "Information interaction")
    IV = ("iv", "Information variation")


class Ordering(Enum):
    Ascending = "Sort patches in ascending rank order"
    Descending = "Sort patches in descending rank order"


class MeasureType(Enum):
    Dist = "Distance measure between two patches"
    STA = "Standalone measure of a patch"


MEASURE_MAP = {
    Measure.MI: mi.mutual_information,
    Measure.JE: entropy.joint_entropy,
    Measure.CE: entropy.conditional_entropy,
    Measure.ENTROPY: entropy.entropy,
    Measure.KL: entropy.kl_divergence,
    Measure.L1: l1_norm.L1Norm,
    Measure.L2: l2_norm.L2Norm,
    Measure.MAX_NORM: max_norm.max_norm,
    Measure.PSNR: psnr.PSNR,
    Measure.SSIM: ssim.SSIM,
    Measure.MI_NORMALIZED: mi.normalized_mutual_information,
    Measure.EI: mi.enigmatic_information,
    Measure.LI: mi.lautum_information,
    Measure.RE: entropy.residual_entropy,
    Measure.MULTI_I: mi.multi_information,
    Measure.BI: mi.binding_information,
    Measure.ELI: mi.exogenous_local_information,
    Measure.IV: mi.information_variation,
    Measure.CROSS_ENTROPY: entropy.cross_entropy,
    Measure.CROSS_ENTROPY_PMF: entropy.cross_entropy_pmf,
    Measure.II: mi.information_interaction,
    Measure.COI: mi.co_information
}


def map_measure_fn(m, measure_type=MeasureType.Dist):
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
    if not isinstance(measure_type, MeasureType):
        raise ValueError(
            "Supplied argument must be an instance of MeasureType Enum")
    if m not in MEASURE_MAP:
        raise ValueError("Measure function %s not found!" % m)

    assert isinstance(measure_type, MeasureType), "Unknown measure type"

    func = MEASURE_MAP[m]

    # Call functions by reflection
    @functools.wraps(func)
    def measure_fn(patches):
        if measure_type == MeasureType.Dist:
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
