
import tensorflow as tf
from pprint import pprint
from cc.map_measure import (Measure, MeasureType, Ordering, map_measure_fn,
                         MEASURE_MAP)
import itertools
import multiprocessing
from joblib import Parallel, delayed
import numpy as np


def _determine_measure_type(measure):
    assert isinstance(measure, Measure)
    if measure in [Measure.JE, Measure.MI, Measure.CE, Measure.L1, Measure.L2,
                   Measure.MAX_NORM, Measure.KL, Measure.SSIM, Measure.PSNR]:
        return MeasureType.Dist
    else:
        return MeasureType.STA


def _swap(p1, p2):
    return p2, p1


def _print(patches):
    for key, value in patches.items():
        print("Current index: %i, Rank: %f" %
              (key, value))


def _sort_patches(patches_data, total_patches, measure, ordering):
    """[summary]

    Arguments:
        patches_data {tensor} -- tensor having shape [number_of_patches, height,width,channel]
        total_patches {int} -- total number of patches 

    Keyword Arguments:
        measure {Measure} -- ranking measure to use for sorting (default: {Measure.JE})
        ordering {Ordering} -- sort order (default: {Ordering.Ascending})
    """
    # TODO - parallel implementation

    measure_type = _determine_measure_type(measure)

    measure_fn = map_measure_fn(measure, measure_type)

    # print("Number of patches: {}".format(total_patches))

    def _compare_numpy(reference_patch, patch):
        patches_to_compare = (reference_patch, patch)
        distance = measure_fn(patches_to_compare)
        return distance

    sess = tf.get_default_session()
    patches_data = sess.run(patches_data)

    if measure_type == MeasureType.STA:
        return _sort_patches_by_content_measure(patches_data, measure_fn, ordering=ordering)

    assert measure_type == MeasureType.Dist, "Supplied measure is not distance measure, please call _sort_patches_by_standalone_measure instead"

    def _swap(i, j):
        # print("Swapping %d with %d" % (i, j))
        patches_data[[i, j]] = patches_data[[j, i]]

    sorted_patches = []
    # debug_sorted_patches = dict()
    distance = -100
    # reference_patch_data = patches_data[0]

    for i in range(0, total_patches):
        # TODO- make configurable
        closest_distance_thus_far = 100
        # print("Closest patch index: %d" % i)
        reference_patch_data = patches_data[i]  # set reference patch
        # sorted_patches.append(reference_patch_data)

        # compare the rest to reference patch
        for j in range(i+1, total_patches):
            # print ("Comparing %d and %d" %(i,j))
            distance = _compare_numpy(reference_patch_data, patches_data[j])
            if j == 1:
                closest_distance_thus_far = distance
                continue
            if ordering == Ordering.Ascending and distance < closest_distance_thus_far:
                closest_distance_thus_far = distance
                _swap(i+1, j)
                # reference_patch_data = patches_data[i]
            elif ordering == Ordering.Descending and distance > closest_distance_thus_far:
                closest_distance_thus_far = distance
                _swap(i+1, j)

    sorted_patches = tf.convert_to_tensor(patches_data, dtype=tf.float32)
    assert sorted_patches.shape[0] == total_patches, "Sorted patches list contains more or less \
    number of patches comparted to original"

    return sorted_patches


def _sort_patches_by_content_measure(patches_data, measure_fn, ordering):
    """[summary]

    Arguments:
        patches_data {np.ndarray} -- the image patches in tensor format
        measure_fn {Measure} -- STA measure function to apply for sorting

    Keyword Arguments:
        ordering {Ordering} -- [description] (default: {Ordering.Ascending})
    """

    assert isinstance(
        patches_data, np.ndarray), "Supplied data must be instance of np.ndarray"

    number_of_patches = patches_data.shape[0]

    def _swap(i, j):
        patches_data[[i, j]] = patches_data[[j, i]]

    sorted_patches = np.array(
        sorted(patches_data, key=lambda patch: measure_fn(patch)))

    assert len(
        sorted_patches) == number_of_patches, "Loss of data when sorting patches data"
    assert patches_data.shape == sorted_patches.shape, "Orignal tensor and sorted tensor have different shapes"

    return tf.convert_to_tensor(sorted_patches, dtype=tf.float32)


def reconstruct_from_patches(patches, image_h, image_w, measure=Measure.MI,ordering=Ordering.Ascending):
    """
    Reconstructs an image from patches of size patch_h x patch_w
    Input: batch of patches shape [n, patch_h, patch_w, patch_ch]
    Output: image of shape [image_h, image_w, patch_ch]

    Arguments:
        patches {tftensor} -- a list of patches of a given image in tftensor or numpy.array format 
        image_h {int} -- image width 
        image_w {int} -- image height 

    Keyword Arguments:
        measure {Measure} -- measure to use to sort patches (default: MI)
    """
    assert patches.shape.ndims == 4, "Patches tensor must be of shape [total_patches, p_w,p_h,c]"

    number_of_patches = patches.shape[0]
    patches = _sort_patches(
        patches, number_of_patches, measure,ordering)

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
