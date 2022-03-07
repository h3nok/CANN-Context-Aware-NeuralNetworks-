import tensorflow as tf

from clo.syllabus_factory import Measure
from clo.syllabus_factory import Ordering
from clo.syllabus_factory import p_por

MEASURE_MAP = {
    'mi': Measure.MI,
    'je': Measure.JE,
    'ce': Measure.CE,
    'e': Measure.ENTROPY,
    'kl': Measure.KL,
    'l1': Measure.L1,
    'l2': Measure.L2,
    'mn': Measure.MAX_NORM,
    'psnr': Measure.PSNR,
    'ssim': Measure.SSIM,
    'min': Measure.MI_NORMALIZED,
    'ei': Measure.EI,
    'li': Measure.LI,
    're': Measure.RE,
    'mui': Measure.MULTI_I,
    'bi': Measure.BI,
    'coi': Measure.COI,
    'cross_entropy': Measure.CROSS_ENTROPY,
    'cep': Measure.CROSS_ENTROPY_PMF,
    'eli': Measure.ELI,
    'ii': Measure.II,
    'iv': Measure.IV
}

ORDERING_MAP = {
    1: Ordering.Ascending,
    0: Ordering.Descending,
    'asc': Ordering.Ascending,
    'des': Ordering.Descending
}


def preprocess_image(image, height, width, is_training=True, measure=Measure.JE,
                     ordering=Ordering.Ascending, patch_size=56):
    assert patch_size <= width
    measure, ordering = decode_measure(measure, ordering)
    assert isinstance(measure, Measure)
    assert isinstance(ordering, Ordering)
    image = tf.image.resize_images(image, [height, width], align_corners=False)
    return p_por(image, height, width, measure, ordering, patch_size, patch_size)


def decode_measure(measure, ordering):
    return MEASURE_MAP[measure], ORDERING_MAP[ordering]
