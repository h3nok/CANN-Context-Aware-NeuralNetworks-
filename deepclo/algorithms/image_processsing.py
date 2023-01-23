from deepclo.core.measures.measure_functions import map_measure_function, \
    Measure, Ordering, determine_measure_classification, MeasureType
import numpy as np


def assess_and_rank_images(
        batch_or_image_blocks,
        content_measure: Measure,
        reference_block_index
):
    """
    Rank image blocks by measuring the content of each block or similarity of each block to
    the reference block

    Args:
        batch_or_image_blocks:
        content_measure:
        reference_block_index:

    Returns:

    """
    ranks = []

    if determine_measure_classification(content_measure) == MeasureType.STANDALONE:
        for i, block in enumerate(batch_or_image_blocks):
            rank = measure_image_content(block, content_measure=content_measure)
            ranks.append(rank)

        return np.array(ranks)

    else:
        if reference_block_index is None:
            raise RuntimeError("Must supply reference image or block.")
        ranks = []
        reference_block = batch_or_image_blocks[reference_block_index]

        for i, block in enumerate(batch_or_image_blocks):
            if i == reference_block_index:
                continue
            rank = measure_content_similarity(reference_block, block, content_measure)
            ranks.insert(i, rank)

        ranks.insert(reference_block_index, 0.0)

        return np.array(ranks)


def measure_image_content(
        image,
        content_measure: Measure
):
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


def measure_content_similarity(
        image1: np.ndarray,
        image2: np.ndarray,
        content_measure: Measure
):
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


def sort_images(
        batch_or_image_blocks: np.array,
        ranks: np.array,
        labels: np.array = np.array([]),
        block_rank_ordering=Ordering.Descending
):
    """
    Sort image blocks using a standalone metric, entropy and ssim

    Args:
        batch_or_image_blocks: np.ndarray of blocks of an image - ordered as in original image
        labels: np.array - labels if input is a batch
        ranks: n.array of blocks ranks
        block_rank_ordering: how blocks or to be ordered - ascending or
        descending order of ranks

    Returns: tuple (sorted blocks, and rank_indices of blocks on original image)
    TODO : Combine with rank_blocks to reduce code
    """
    ranks_indices = None
    assert batch_or_image_blocks.shape[0] == ranks.shape[0]
    if block_rank_ordering in Ordering.Ascending.value:
        ranks_indices = ranks.argsort()
    elif block_rank_ordering in Ordering.Descending.value:
        ranks_indices = (-ranks).argsort()

    if labels.size != 0:
        return batch_or_image_blocks[ranks_indices], labels[ranks_indices], ranks_indices

    return batch_or_image_blocks[ranks_indices], ranks_indices


def blocks_to_3d_volume(
        blocks: np.ndarray,
        image_height: int,
        image_width: int,
        number_of_channels: int,
        stride: int = None
):
    pass


def blocks_to_2d_image(
        blocks: np.ndarray,
        image_height: int,
        image_width: int,
        number_of_channels: int,
        stride: int = None
):
    """
       Reconstruct a 2-d image from all patches.
       Patches are assumed to be square and overlapping depending on the stride. The image is constructed
       by filling in the patches from left to right, top to bottom, averaging the overlapping parts.
    Parameters
    -----------

    blocks: 4D ndarray with shape (patch_number,patch_height,patch_width,channels)
        Array containing extracted patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.
    image_height: int
        original height of image to be reconstructed
    image_width: int
        original width of image to be reconstructed
    number_of_channels: int
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
    rows = ((image_height - patch_size) // stride) * stride + patch_size
    cols = ((image_width - patch_size) // stride) * stride + patch_size

    if number_of_channels == 1:
        reconstructed_img_array = np.zeros((rows, cols))
        divide_image = np.zeros((rows, cols))
    else:
        reconstructed_img_array = np.zeros((rows, cols, number_of_channels))
        divide_image = np.zeros((rows, cols, number_of_channels))

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
