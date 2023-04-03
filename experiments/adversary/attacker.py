import numpy as np
from deepclo.pipe.dataset import ImageDataProvider
import matplotlib.pyplot as plt


def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 224, 224)):
    applied_patch = np.zeros(image_size)
    if mask_type == 'rectangle' or mask_type == 'square':
        '''Square Patch
        # patch rotation
        rotation_angle = np.random.choice(4)
        for i in range(patch.shape[0]):
            patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
        '''
        # patch location
        x_location, y_location = np.random.randint(low=0, high=image_size[1] - patch.shape[1]), \
                                 np.random.randint(
                                     low=0, high=image_size[2] - patch.shape[2])
        for i in range(patch.shape[0]):
            applied_patch[:, x_location:x_location + patch.shape[1],
            y_location:y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()

    mask[mask != 0] = 1.0

    return applied_patch, mask, x_location, y_location


class Attacker:
    def __init__(self, img: np.ndarray):
        self._img = img

    def one_pixel_attack(self, xs):
        """

        Args:
            xs: # pixel position and color = x,y,r,g,b

        Returns:

        """
        # If this function is passed just one perturbation vector,
        # pack it in a list to keep the computation the same
        if xs.ndim < 2:
            xs = np.array([xs])

        # Copy the image n == len(xs) times so that we can
        # create n new perturbed images
        tile = [len(xs)] + [1] * (xs.ndim + 1)
        imgs = np.tile(self._img, tile)

        # Make sure to floor the members of xs as int types
        xs = xs.astype(int)

        for x, img in zip(xs, imgs):
            # Split x into an array of 5-tuples (perturbation pixels)
            # i.e., [[x,y,r,g,b], ...]
            pixels = np.split(x, len(x) // 5)
            for pixel in pixels:
                # At each pixel's x,y position, assign its rgb value
                x_pos, y_pos, *rgb = pixel
                img[x_pos, y_pos] = rgb

        return imgs

    @staticmethod
    def patch_attack(patch_type='rectangle', image_size=(3, 224, 224), noise_percentage=0.03):
        if patch_type == 'rectangle':
            # Rectangular Patch
            mask_length = int(((noise_percentage) * image_size[1] * image_size[2]) ** 0.45)
            mask_width = int(((noise_percentage) * image_size[1] * image_size[2]) ** 0.56)
            patch = np.random.rand(image_size[0], mask_length, mask_width)
        elif patch_type == "square":
            # Square Patch
            mask_length = int((noise_percentage * image_size[1] * image_size[2]) ** 0.5)
            patch = np.random.rand(image_size[0], mask_length, mask_length)

        return patch


def plot_image(image, label_true=None, class_names=None, label_pred=None):
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    plt.grid()
    plt.imshow(image.astype(np.uint8))

    # Show true and predicted classes
    if label_true is not None and class_names is not None:
        labels_true_name = class_names[label_true]
        if label_pred is None:
            xlabel = "True: " + labels_true_name
        else:
            # Name of the predicted class
            labels_pred_name = class_names[label_pred]

            xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name

        # Show the class on the x-axis
        plt.xlabel(xlabel)

    plt.xticks([])  # Remove ticks from the plot
    plt.yticks([])
    plt.show()  # Show the plot


ds = ImageDataProvider(dataset_name='cifar10')
# assert ds.x_train.shape == (50000, 32, 32, 3)
assert ds.input_shape == (32, 32, 3)
image_id = 99
# plot_image(ds.x_test[image_id])
attacker = Attacker(ds.x_test[image_id])
patch = attacker.patch_attack()
print(patch.shape)
applied_patch, mask, x_location, y_location = mask_generation(patch=patch, image_size=(3, 224, 224))
applied_patch = np.moveaxis(applied_patch, 0, -1)
plot_image(applied_patch)

#
# attack_vectors = [
#     np.array([4, 4, 255, 255, 0]),
#     np.array([30, 4, 255, 255, 0]),
#     np.array([4, 30, 255, 255, 0]),
#     np.array([30, 30, 255, 255, 0]),
#     np.array([16, 16, 255, 255, 0])]
# # pixel = y, x, r,g,b
#
# for pixel in attack_vectors:
#     image_perturbed = attacker.one_pixel_attack(pixel)
#     plot_image(image_perturbed[0])
