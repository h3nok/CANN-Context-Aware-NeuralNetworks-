import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


class ImageTo3DVolume:
    """
    A class to convert a given image into a 3D volume by splitting it into equal-sized patches and ranking them based on a specified metric.

    Attributes:
        _image (numpy array): A numpy array representing the input image.
        _patch_size (tuple): A tuple of integers representing the width and height of each patch.
        _ranking_metric (str): A string specifying the ranking metric for the patches.
        _reference_patch (numpy array): A numpy array representing the reference patch for distance-based metrics.
    """

    def __init__(self, image=None, patch_size=(8, 8), ranking_metric='entropy', reference_patch=None):
        self._image = image
        self._patch_size = patch_size
        self._ranking_metric = ranking_metric
        self._reference_patch = reference_patch

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Image must be a NumPy array.")
        self._image = value

    @property
    def patch_size(self):
        return self._patch_size

    @patch_size.setter
    def patch_size(self, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("Patch size must be a tuple of two integers.")
        self._patch_size = value

    @property
    def ranking_metric(self):
        return self._ranking_metric

    @ranking_metric.setter
    def ranking_metric(self, value):
        if not isinstance(value, str):
            raise ValueError("Ranking metric must be a string.")
        self._ranking_metric = value

    @property
    def reference_patch(self):
        return self._reference_patch

    @reference_patch.setter
    def reference_patch(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("Reference patch must be a NumPy array or None.")
        self._reference_patch = value

    def split_image_into_patches(self, img):
        height, width, _ = img.shape
        patch_height, patch_width = self._patch_size
        patches = []

        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                patch = img[y:y + patch_height, x:x + patch_width]
                if patch.shape == (patch_height, patch_width, 3):
                    patches.append(patch)
        return patches

    def rank_patches(self, patches):
        if self._ranking_metric == 'entropy':
            ranked_patches = sorted(patches, key=lambda p: entropy(p.ravel()))
        elif self._ranking_metric in ['l1_norm', 'mutual_info']:
            if self._reference_patch is None:
                self._reference_patch = min(patches, key=lambda p: entropy(p.ravel()))
            if self._ranking_metric == 'l1_norm':
                ranked_patches = sorted(patches, key=lambda p: np.linalg.norm(p.ravel() - self._reference_patch.ravel(), ord=1))
            else:
                ranked_patches = sorted(patches, key=lambda p: -mutual_info_score(p.ravel(), self._reference_patch.ravel()))
        else:
            raise ValueError(f"Invalid ranking metric: {self._ranking_metric}")
        return ranked_patches

    @staticmethod
    def create_3d_volume(ranked_patches):
        volume = np.stack(ranked_patches, axis=-1)
        return volume

    def preprocess_image(self):
        """
        Preprocess the input image by splitting it into patches, ranking the patches, and then creating a 3D volume.

        Returns:
            numpy array: A 3D numpy array representing the volume.
        """
        if self._image is None:
            raise ValueError("Image is not set.")

        patches = self.split_image_into_patches(self._image)
        ranked_patches = self.rank_patches(patches)
        volume = self.create_3d_volume(ranked_patches)

        return volume
