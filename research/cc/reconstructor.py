
import tensorflow as tf
import numpy as np

try:
    from cc.map_measure import (Measure, MeasureType,
                                Ordering, map_measure_fn,
                                MEASURE_MAP)
    from cc.cc_utils import ConfigureLogger
except ImportError as error:
    print(error)
    from map_measure import (Measure, MeasureType,
                             Ordering, map_measure_fn,
                             MEASURE_MAP)
    from cc_utils import ConfigureLogger

_logger = ConfigureLogger(__file__, '.')


def _determine_measure_type(measure):
    print(type(measure))
    assert isinstance(measure, Measure), "{} not found.".format(measure.value)
    if measure in [Measure.JE, Measure.MI, Measure.CE, Measure.L1, Measure.L2,
                   Measure.MAX_NORM, Measure.KL, Measure.SSIM, Measure.PSNR,
                   Measure.MI_NORMALIZED, Measure.LI, Measure.IV, Measure.CROSS_ENTROPY,
                   Measure.CROSS_ENTROPY_PMF]:

        return MeasureType.Dist
    else:
        return MeasureType.STA


def _swap(p1, p2):
    return p2, p1


def _print(patches):
    for key, value in patches.items():
        print("Current index: %i, Rank: %f" %
              (key, value))


def sort_patches_or_images(patches_data, total_patches, measure, ordering, curriculum=False, labels=None):
    """[summary]

    Arguments:
        patches_data {tensor} -- tensor having shape [number_of_patches, height,width,channel]
        total_patches {int} -- total number of patches

    Keyword Arguments:
        measure {Measure} -- ranking measure to use for sorting (default: {Measure.JE})
        ordering {Ordering} -- sort order (default: {Ordering.Ascending})
    """
    coord = tf.train.Coordinator()
    # TODO - parallel implementation
    _logger.info("Entering _sort_patches ... ")
    measure_type = _determine_measure_type(measure)

    measure_fn = map_measure_fn(measure, measure_type)

    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess, coord=coord)
    patches_data = sess.run(patches_data)
    labels_data = None
    if curriculum:
        if labels is not None:
            labels_data = sess.run(labels)
    sess.close()

    if measure_type == MeasureType.STA:
        _logger.info(
            'Measure type is standalone, calling _sort_patches_by_content_measure ...')
        return sort_by_content_measure(patches_data, measure_fn, labels)

    if measure_type != MeasureType.Dist:
        _logger.error(
            "Supplied measure is not distance measure, please call _sort_patches_by_standalone_measure instead")

    _logger.info(
        "Sorting patches by distance measure, measure: {}".format(measure.value))

    def _compare_numpy(reference_patch, patch):
        patches_to_compare = (reference_patch, patch)
        distance = measure_fn(patches_to_compare)
        return distance

    def _swap(i, j):
        # print("Swapping %d with %d" % (i, j))
        patches_data[[i, j]] = patches_data[[j, i]]

    def _swap_labels(i, j):
        labels_data[[i, j]] = labels_data[[j, i]]

    for i in range(0, total_patches):
            # TODO- make configurable
        closest_distance_thus_far = 100
        reference_patch_data = patches_data[i]  # set reference patch
        # sorted_patches.append(reference_patch_data)

        # compare the rest to reference patch
        for j in range(i+1, total_patches):
                # print ("Comparing %d and %d" %(i,j))
            distance = _compare_numpy(
                reference_patch_data, patches_data[j])
            if j == 1:
                closest_distance_thus_far = distance
                continue
            if ordering == Ordering.Ascending and distance < closest_distance_thus_far:
                closest_distance_thus_far = distance
                _swap(i+1, j)
                if curriculum:
                    _swap_labels(i+1, j)
                # reference_patch_data = patches_data[i]
            elif ordering == Ordering.Descending and distance > closest_distance_thus_far:
                closest_distance_thus_far = distance
                _swap(i+1, j)
                if labels is not None:
                    _swap_labels(i+1, j)

    sorted_patches = tf.convert_to_tensor(patches_data, dtype=tf.float32)
    sorted_labels = None
    if curriculum:
        sorted_labels = tf.convert_to_tensor(labels_data, tf.uint8)
        assert sorted_labels.shape[0] == total_patches

    assert sorted_patches.shape[0] == total_patches, _logger.error("Sorted patches list contains more or less \
        number of patches compared to original")
    _logger.info(
        "Successfully sorted patches, closing session and exiting ...")

    return sorted_patches, sorted_labels


def sort_by_content_measure(patches_data, measure_fn, labels, curriculum=False):
    """[summary]

    Arguments:
        patches_data {np.ndarray} -- the image patches in tensor format
        measure_fn {Measure} -- STA measure function to apply for sorting

    Keyword Arguments:
        ordering {Ordering} -- [description] (default: {Ordering.Ascending})
    """

    _logger.info("Entering sort patches by content measure ...")

    assert isinstance(
        patches_data, np.ndarray), "Supplied data must be instance of np.ndarray"

    number_of_patches = patches_data.shape[0]

    sorted_patches = None
    sorted_labels = None

    if not curriculum:
        sorted_patches = np.array(
            sorted(patches_data, key=lambda patch: measure_fn(patch)))
    else:
        patches_data = zip(patches_data, labels)
        sorted_patches, sorted_labels = np.array(zip(*sorted(patches_data,
                                                             key=lambda patch: measure_fn(patches_data[patch]))))

    assert len(
        sorted_patches) == number_of_patches, \
        _logger.error("Loss of data when sorting patches data")
    assert patches_data.shape == sorted_patches.shape, \
        _logger.error("Original tensor and sorted tensor have different shapes")

    _logger.info("Successfully sorted patches using content measure ")

    sorted_labels = tf.convert_to_tensor(sorted_labels, dtype=tf.int8)
    sorted_patches = tf.convert_to_tensor(sorted_patches, dtype=tf.float32)

    if curriculum:
        return sorted_patches, sorted_labels
    else:
        return sorted_patches


def reconstruct_from_patches(patches, image_h, image_w, measure=Measure.MI, ordering=Ordering.Ascending):
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
    assert patches.shape.ndims == 4, _logger.error(
        "Patches tensor must be of shape [total_patches, p_w,p_h,c]")

    number_of_patches = patches.shape[0]
    _logger.info("Entering reconstruct_from_patches, number of patches: {}".format(
        number_of_patches))
    patches,_ = sort_patches_or_images(
        patches, number_of_patches, measure, ordering)

    _logger.info("Reconstructing sample ...")
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

    _logger.info(
        "Successfully reconstructed image, measure: {}".format(measure.value))

    return image[0]
