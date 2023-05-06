import numpy as np
from skimage import data
import matplotlib.pyplot as plt


def visualize_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 15))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.show()


class PixelAttackGenerator:
    def __init__(self, n, perturbation=128):
        """
        Initialize the PixelAttackGenerator with a specified number of pixels to perturb.

        :param n: The number of pixels to perturb
        :param perturbation: The perturbation value added to each selected pixel (default: 128)
        """
        self.n = n
        self.perturbation = perturbation

    def generate_attack(self, image, quadrant=None):
        """
        Generate an n-pixel adversarial attack on an input image, localized to a specific quadrant.

        :param image: A NumPy array representing the input image of shape (height, width, channels)
        :param quadrant: The target quadrant for the attack (1, 2, 3, or 4), or None for the whole image
        :return: A NumPy array representing the perturbed image
        """
        perturbed_image = image.copy()
        height, width, channels = image.shape

        if quadrant is not None:
            half_height, half_width = height // 2, width // 2

            if quadrant == 1:
                y_min, y_max, x_min, x_max = 0, half_height, 0, half_width
            elif quadrant == 2:
                y_min, y_max, x_min, x_max = 0, half_height, half_width, width
            elif quadrant == 3:
                y_min, y_max, x_min, x_max = half_height, height, 0, half_width
            elif quadrant == 4:
                y_min, y_max, x_min, x_max = half_height, height, half_width, width
            else:
                raise ValueError("Invalid quadrant value. Must be 1, 2, 3, or 4.")
        else:
            y_min, y_max, x_min, x_max = 0, height, 0, width

        pixel_coordinates = np.random.choice((y_max - y_min) * (x_max - x_min), self.n, replace=False)
        pixel_coordinates = np.array([np.unravel_index(i, (y_max - y_min, x_max - x_min)) for i in pixel_coordinates])
        pixel_coordinates[:, 0] += y_min
        pixel_coordinates[:, 1] += x_min

        for y, x in pixel_coordinates:
            for c in range(channels):
                perturbed_image[y, x, c] = np.clip(perturbed_image[y, x, c] + self.perturbation, 0, 255)

        return perturbed_image

    def random_perturbation(self, image, perturb_range=(-10, 10)):
        perturbed_image = image.copy()
        indices = np.random.randint(0, image.shape[0], (self.n, 2))
        for idx in indices:
            perturb_value = np.random.randint(perturb_range[0], perturb_range[1])
            perturbed_image[idx[0], idx[1]] = np.clip(perturbed_image[idx[0], idx[1]] + perturb_value, 0, 255)
        return perturbed_image

    def directional_perturbation(self, image, perturb_value=5, increase=True):
        perturbed_image = image.copy()
        indices = np.random.randint(0, image.shape[0], (self.n, 2))
        for idx in indices:
            if increase:
                perturbed_image[idx[0], idx[1]] = np.clip(perturbed_image[idx[0], idx[1]] + perturb_value, 0, 255)
            else:
                perturbed_image[idx[0], idx[1]] = np.clip(perturbed_image[idx[0], idx[1]] - perturb_value, 0, 255)
        return perturbed_image

    @staticmethod
    def generate_patch_attack(image, patch, x, y):
        """
        Generate a patch adversarial attack on an input image by overlaying a patch at a specific location.

        :param image: A NumPy array representing the input image of shape (height, width, channels)
        :param patch: A NumPy array representing the adversarial patch of shape (patch_height, patch_width, channels)
        :param x: The x-coordinate of the top-left corner of the patch in the image
        :param y: The y-coordinate of the top-left corner of the patch in the image
        :return: A NumPy array representing the perturbed image
        """
        perturbed_image = image.copy()
        patch_height, patch_width, _ = patch.shape
        perturbed_image[y:y + patch_height, x:x + patch_width] = patch
        return perturbed_image

    @staticmethod
    def generate_spatial_attack(image, transformation_matrix):
        """
        Generate a spatial adversarial attack on an input image using an affine transformation matrix.

        :param image: A NumPy array representing the input image of shape (height, width, channels)
        :param transformation_matrix: A 3x3 NumPy array representing the affine transformation matrix
        :return: A NumPy array representing the perturbed image
        """
        from skimage.transform import warp

        perturbed_image = warp(image, transformation_matrix)
        return perturbed_image


# Load an example image
image = data.astronaut()

# Create instances of the attacks
n_pixel_attack = PixelAttackGenerator(n=50)
patch = np.zeros((50, 50, 3), dtype=np.uint8)
spatial_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Apply the attacks
perturbed_n_pixel = n_pixel_attack.generate_attack(image, quadrant=1)
perturbed_patch = n_pixel_attack.generate_patch_attack(image, patch, x=50, y=50)
perturbed_spatial = n_pixel_attack.generate_spatial_attack(image, spatial_transform)

# Visualize the results
visualize_images([image, perturbed_n_pixel,
                  perturbed_patch, perturbed_spatial],
                 ['Original Image', 'N-Pixel Attack',
                  'Patch Attack', 'Spatial Attack'])
