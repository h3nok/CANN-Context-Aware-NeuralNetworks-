from deepclo.core.measures.measure_functions import map_measure_function, \
    Measure, Ordering, determine_measure_classification, MeasureType
import numpy as np


def measure_image_content(image, content_measure: Measure):
    """
    Rank images - using standalone metric
    Standalone measures –measure some characteristic
    of a patch. For example, the peak signal-to-noise ratio measure
    returns a ratio between maximum useful signal to the amount
    of noise present in a patch.

    Args:
        image: np.ndarray
        content_measure:

    Returns:

    """
    content_measure_fn = map_measure_function(content_measure)
    result = content_measure_fn(image)

    return result


def measure_content_similarity(image1: np.ndarray, image2: np.ndarray, content_measure: Measure):
    """
    Similarity measures – these measures on the other
    hand, compare a pair of patches. The comparison measures can
    be measures of similarity or dissimilarity like L1-norm and
    structural similarity or information-theoretic-measures that
    compare distribution of pixel values such as joint entropy.

    Args:
        image1:
        image2:
        content_measure:

    Returns:

    """
    assert content_measure
    assert determine_measure_classification(content_measure) == MeasureType.DISTANCE
    content_measure_fn = map_measure_function(content_measure)
    result = content_measure_fn([image1, image2])

    return result


def sort_image_blocks(blocks: np.array,
                      ranks: np.array,
                      block_rank_ordering=Ordering.Descending):
    """
    Sort image blocks using a standalone metric, entropy and ssim

    Args:
        blocks: np.ndarray of blocks of an image - ordered as in original image
        ranks: n.array of blocks ranks
        block_rank_ordering: how blocks or to be ordered - ascending or
        descending order of ranks

    Returns: tuple (sorted blocks, and rank_indices of blocks on original image)
    TODO : Combine with rank_blocks to reduce code
    """
    ranks_indices = None
    assert blocks.shape[0] == ranks.shape[0]
    if block_rank_ordering in Ordering.Ascending.value:
        ranks_indices = ranks.argsort()
    elif block_rank_ordering in Ordering.Descending.value:
        ranks_indices = (-ranks).argsort()

    return blocks[ranks_indices], ranks_indices


def construct_new_input(blocks: np.ndarray, im_h: int, im_w: int, n_channels: int, stride: int = None):
    """
       Reconstruct the image from all patches.
       Patches are assumed to be square and overlapping depending on the stride. The image is constructed
       by filling in the patches from left to right, top to bottom, averaging the overlapping parts.
    Parameters
    -----------
    blocks: 4D ndarray with shape (patch_number,patch_height,patch_width,channels)
        Array containing extracted patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    im_h: int
        original height of image to be reconstructed
    im_w: int
        original width of image to be reconstructed
    n_channels: int
        number of channels the image has. For  RGB image, n_channels = 3
    stride: int
           desired patch stride
    Returns
    -----------
    reconstructedim: ndarray with shape (height, width, channels)
                      or ndarray with shape (height, width) if output image only has one channel
                    Reconstructed image from the given patches
    """

    patch_size = blocks.shape[1]  # patches assumed to be square
    if not stride:
        stride = patch_size
    # Assign output image shape based on patch sizes
    rows = ((im_h - patch_size) // stride) * stride + patch_size
    cols = ((im_w - patch_size) // stride) * stride + patch_size

    if n_channels == 1:
        reconstructed_img_array = np.zeros((rows, cols))
        divide_image = np.zeros((rows, cols))
    else:
        reconstructed_img_array = np.zeros((rows, cols, n_channels))
        divide_image = np.zeros((rows, cols, n_channels))

    number_patches_per_row = (cols - patch_size + stride) / stride  # number of patches needed to fill out a row
    total_patches = blocks.shape[0]
    assert total_patches == number_patches_per_row ** 2
    init_row, init_col = 0, 0

    # extract each patch and place in the zero matrix and sum it with existing pixel values

    # fill out top left corner using first patch
    reconstructed_img_array[init_row:patch_size, init_col:patch_size] = blocks[0]
    divide_image[init_row:patch_size, init_col:patch_size] = np.ones(blocks[0].shape)

    patch_num = 1

    while patch_num <= total_patches - 1:

        init_col = init_col + stride
        reconstructed_img_array[init_row:init_row + patch_size, init_col:patch_size + init_col] += blocks[patch_num]
        divide_image[init_row:init_row + patch_size, init_col:patch_size + init_col] += np.ones(blocks[patch_num].shape)

        if np.remainder(patch_num + 1, number_patches_per_row) == 0 and patch_num < total_patches - 1:
            init_row = init_row + stride
            init_col = 0
            reconstructed_img_array[init_row:init_row + patch_size, init_col:patch_size] += blocks[patch_num + 1]
            divide_image[init_row:init_row + patch_size, init_col:patch_size] += np.ones(blocks[patch_num].shape)
            patch_num += 1
        patch_num += 1

    # Average out pixel values
    reconstructedim = reconstructed_img_array / divide_image

    return reconstructedim.astype(np.uint8)

