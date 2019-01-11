
import tensorflow as tf
from map_measure import measure_map, Measure, map_measure_fn, MeasureType


def _determine_measure_type(measure):
    assert isinstance(measure, Measure)
    if measure in [Measure.JE, Measure.MI, Measure.CE]:
        return MeasureType.Dist


def _sort_patches_by_distance_measure(patches_data, total_patches, measure=Measure.JE, ordering=0):
    # TODO - parallel implementation

    measure_type = _determine_measure_type(measure)

    assert measure_type == MeasureType.Dist, "Supplied measure is not distance measure, please call _sort_patches_by_standalone_measure instead"

    measure_fn = map_measure_fn(measure, measure_type)
    patches_to_compare = None
    ranked_patches = dict()
    rank = None

    for i in range(total_patches):
        if measure_type == MeasureType.Dist:
            patches_to_compare = (patches_data[i], patches_data[i+1])
        else:
            patches_to_compare = patches_data[i]
    # measure_fn(patches_to_compare)


def reconstruct_from_patches(patches, image_h, image_w, measure=Measure.JE):
    """
    Reconstructs an image from patches of size patch_h x patch_w
    Input: batch of patches shape [n, patch_h, patch_w, patch_ch]
    Output: image of shape [image_h, image_w, patch_ch]

    Arguments:
        patches {[type]} -- [description]
        image_h {[type]} -- [description]
        image_w {[type]} -- [description]

    Keyword Arguments:
        measure {[type]} -- [description] (default: {None})
    """
    assert patches.shape.ndims == 4, "Patches tensor must be of shape [total_patches, p_w,p_h,c]"

    number_of_patches = patches.shape[0]
    _sort_patches_by_distance_measure(patches, number_of_patches, measure)

    pad = [[0, 0], [0, 0]]
    patch_h = patches.shape[1].value
    patch_w = patches.shape[2].value
    patch_ch = patches.shape[3].value
    p_area = patch_h * patch_w
    h_ratio = image_h // patch_h
    w_ratio = image_w // patch_w

    image = tf.reshape(patches, [1, h_ratio, w_ratio, p_area, patch_ch])
    image = tf.split(image, p_area, 3)
    image = tf.stack(image, 0)
    image = tf.reshape(image, [p_area, h_ratio, w_ratio, patch_ch])
    image = tf.batch_to_space_nd(image, [patch_h, patch_w], pad)

    return image[0]
