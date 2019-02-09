from cc.patch_proposals import generate_patches_v2, p_por
from cc.reconstructor import reconstruct_from_patches
from cc.map_measure import Measure
from cc.map_measure import Ordering
import tensorflow as tf
import numpy as np 


def preprocess_image(image, height, width, is_training=True, measure=Measure.JE, ordering=Ordering.Ascending, patch_size=56):
    return p_por(image, height, width, measure, ordering, patch_size, patch_size)
