import numpy as np
from numpy.linalg import norm
from numpy import inf
from skimage.metrics import structural_similarity


def l1_norm(patch_1, patch_2):
    """
    L1 Norm (Manhattan distance or Taxicab norm) is the sum of the magnitudes
    of the vectors in a space. It is the most natural way of measure distance
    between vectors, that is the sum of absolute difference of the components
    of the vectors. In this norm, all the components of the vector are weighted
    equally.
    https://en.wikipedia.org/wiki/Norm_(mathematics)

    Args:
        patch_1: nxm numpy array representing patch 1
        patch_2: nxm numpy array representing patch 2

    Returns: 0.0 if the two patches are identical, positive float otherwise

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"

    if np.array_equal(patch_1, patch_2):
        print("Warning: Patches are binary equivalent, L1 Norm = 0.0")
        return 0.0

    # combine the two tensors into one
    patch_data = np.concatenate((patch_1, patch_2)).flatten()
    return round(norm(patch_data, 1), 4)


def l2_norm(patch_1, patch_2):
    """
    Is the most popular norm, also known as the Euclidean norm.
    It is the shortest distance to go from one point to another.
    https://en.wikipedia.org/wiki/Norm_(mathematics)

    Args:
        patch_1: nxm numpy array representing patch 1
        patch_2: nxm numpy array representing patch 2

    Returns: positive float, l2 norm of the two patches otherwise 0

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, L2 Norm = 0.0"

    # combine the two tensors into one
    patch_data = np.concatenate((patch_1, patch_2)).flatten()
    return round(norm(patch_data, 2), 4)


def max_norm(patch_1, patch_2):
    """
    https://en.wikipedia.org/wiki/Norm_(mathematics)

    Args:
        patch_1:
        patch_2:

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), "Patches are binary equivalent, Distance = 0"

    # combine the two tensors into one
    patch_data = np.concatenate((patch_1, patch_2)).flatten()
    return round(norm(patch_data, inf), 4)


def psnr(patch_1, patch_2, maximum_data_value=255, ignore=None):
    """
    title::
       psnr

    description::
       This method will compute the peak-signal-to-noise ratio (PSNR) between
       two provided data sets.  The PSNR will be computed for the ensemble
       data.  If the PSNR is desired for a particular slice of the provided
       data, then the data sets provided should represent those slices.

    attributes::
       dataset1
          An array-like object containing the first data set.
       dataset2
          An array-like object containing the second data set.
       maximumDataValue
          The maximum value that might be contained in the data set (this
          is not necessarily the maximum value in the data set, but
          rather it is the largest value that any member of the data set
          might take on).
          -- RGB is default
       ignore
          A scalar value that will be ignored in the data sets.  This can
          be used to mask data in the provided data set from being
          included in the analysis. This value will be looked for in both
          of the provided data sets, and only an intersection of positions
          in the two data sets will be included in the computation. [default
          is None]

    author::
       Carl Salvaggio

    copyright::
       Copyright (C) 2015, Rochester Institute of Technology

    license::
       GPL

    version::
       1.0.0

    disclaimer::
       This source code is provided "as is" and without warranties as to
       performance or merchantability. The author and/or distributors of
       this source code may have made statements about this source code.
       Any such statements do not constitute warranties and shall not be
       relied on by the user in deciding whether to use this source code.

       This source code is provided without any express or implied warranties
       whatsoever. Because of the diversity of conditions and hardware under
       which this source code may be used, no warranty of fitness for a
       particular purpose is offered. The user is advised to test the source
       code thoroughly before relying on it. The user must assume the entire
       risk of using the source code.
    """

    # Make sure that the provided data sets are numpy ndarrays, if not
    # convert them and flatten te data sets for analysis
    if type(patch_1).__module__ != np.__name__:
        d1 = np.asarray(patch_1).flatten()
    else:
        d1 = patch_1.flatten()

    if type(patch_2).__module__ != np.__name__:
        d2 = np.asarray(patch_2).flatten()
    else:
        d2 = patch_2.flatten()

    # Make sure that the provided data sets are the same size
    if d1.size != d2.size:
        raise ValueError('Provided pipe must have the same size/shape')

    # Check if the provided data sets are identical, and if so, return an
    # infinite peak-signal-to-noise ratio
    if np.array_equal(d1, d2):
        return float('inf')

    # If specified, remove the values to ignore from the analysis and compute
    # the element-wise difference between the data sets
    if ignore is not None:
        index = np.intersect1d(np.where(d1 != ignore)[0],
                               np.where(d2 != ignore)[0])
        error = d1[index].astype(np.float64) - \
                d2[index].astype(np.float64)
    else:
        error = d1.astype(np.float64) - d2.astype(np.float64)

    # Compute the mean-squared error
    mean_squared_error = np.sum(error ** 2) / error.size

    # Return the peak-signal-to-noise ratio
    return round(10.0 * np.log10(maximum_data_value ** 2 / mean_squared_error), 4)


def ssim(patch_1, patch_2):
    """
    The Structural Similarity Index (SSIM) is a perceptual metric
    that quantifies image quality degradation* caused by processing
    such as data compression or by losses in data transmission. It
    is a full reference metric that requires two images from the same
    image capture— a reference image and a processed image.

    Args:
        patch_1:
        patch_2:

    Returns:

    """
    assert isinstance(patch_1, np.ndarray), "Patch data must be a numpy array."
    assert isinstance(patch_2, np.ndarray), "Patch data must be a numpy array."
    assert patch_1.shape == patch_2.shape, "Patches must have similar tensor shapes of [p_w, p_h, c]"
    assert not np.array_equal(
        patch_1, patch_2), \
        "Patches are binary equivalent, Distance = 0"

    patch_1 = patch_1.flatten()
    patch_2 = patch_2.flatten()

    value = structural_similarity(patch_1, patch_2)
    return round(value, 4)


def dssim(patch_1, patch_2):
    return (1 - ssim(patch_1=patch_1, patch_2=patch_2)) / 2