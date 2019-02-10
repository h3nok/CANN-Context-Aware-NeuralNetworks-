from cc.patch_proposals import generate_patches_v2, p_por
from cc.reconstructor import reconstruct_from_patches
from cc.map_measure import Measure
from cc.map_measure import Ordering
import tensorflow as tf
import numpy as np

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
    'ssim': Measure.SSIM
}

ORDERING_MAP = {
    1: Ordering.Ascending,
    0: Ordering.Descending
}


def preprocess_image(image, height, width, is_training=True, measure=Measure.JE, ordering=Ordering.Ascending, patch_size=56):
    assert patch_size <= width
    assert isinstance(measure, Measure)
    assert isinstance(ordering, Ordering)
    image = tf.image.resize_images(image, [height, width], align_corners=False)
    return p_por(image, height, width, measure, ordering, patch_size, patch_size)


def decode_measure(measure, ordering):
    return MEASURE_MAP[measure], ORDERING_MAP[ordering]
