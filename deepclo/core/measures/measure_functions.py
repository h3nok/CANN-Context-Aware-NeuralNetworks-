import functools
from enum import Enum

from deepclo.core.measures import information_theory as entropy
from deepclo.core.measures.statistical import l1_norm, l2_norm, max_norm, psnr, ssim
from deepclo.utils import configure_logger

_logger = configure_logger(__file__, '../../')


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
    CROSS_ENTROPY = ("cross_entropy", "Cross _entropy")
    CROSS_ENTROPY_PMF = ("cep", "Cross _entropy pmf")
    ELI = ("eli", "Exogenous local information")
    II = ("ii", "Information interaction")
    IV = ("iv", "Information variation")
    IQ = ('iq', "Image quality based on FFT spectrum analysis")


class Ordering(Enum):
    Ascending = [1, 'asc']
    Descending = [0, 'desc']


class MeasureType(Enum):
    DISTANCE = "Distance measure between two patches"
    STANDALONE = "Standalone measure of a patch"


MEASURE_MAP = {
    Measure.MI: entropy.mutual_information,
    Measure.JE: entropy.joint_entropy,
    Measure.CE: entropy.conditional_entropy,
    Measure.ENTROPY: entropy.entropy,
    Measure.KL: entropy.kl_divergence,
    Measure.L1: l1_norm,
    Measure.L2: l2_norm,
    Measure.MAX_NORM: max_norm,
    Measure.PSNR: psnr,
    Measure.SSIM: ssim,
    Measure.MI_NORMALIZED: entropy.normalized_mutual_information,
    Measure.EI: entropy.enigmatic_information,
    Measure.LI: entropy.lautum_information,
    Measure.RE: entropy.residual_entropy,
    Measure.MULTI_I: entropy.multi_information,
    Measure.BI: entropy.binding_information,
    Measure.ELI: entropy.exogenous_local_information,
    Measure.IV: entropy.information_variation,
    Measure.CROSS_ENTROPY: entropy.cross_entropy,
    Measure.CROSS_ENTROPY_PMF: entropy.cross_entropy_pmf,
    Measure.II: entropy.information_interaction,
    Measure.COI: entropy.co_information
}


def determine_measure_classification(m):
    if m in [Measure.ENTROPY, Measure.SSIM]:
        return MeasureType.STANDALONE

    return MeasureType.DISTANCE


def map_measure_function(m, measure_type=None):
    """[summary]

    Arguments:
        m {Measure} -- measure to use

    Keyword Arguments:
        measureType {MeasureType} -- measure type (default: {MeasureType.Dist})
    """
    if not measure_type:
        measure_type = determine_measure_classification(m)

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
    def measure_fn(image_patches):
        if measure_type == MeasureType.DISTANCE:
            return func(image_patches[0], image_patches[1])
        else:
            return func(image_patches)

    return measure_fn
