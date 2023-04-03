import numpy as np
import random
from PIL import Image


class PatchAttackGenerator:
    def __init__(self, pregenerated_patches, patch_size, attack_intensity):
        self.patches = pregenerated_patches
        self.patch_size = patch_size
        self.attack_intensity = attack_intensity

    def apply_patch(self, image):
        image = Image.fromarray(image.astype(np.uint8))
        img_w, img_h = image.size
        p_w, p_h = self.patch_size

        for _ in range(self.attack_intensity):
            patch = random.choice(self.patches)
            x = random.randint(0, img_w - p_w)
            y = random.randint(0, img_h - p_h)
            image.paste(patch, (x, y))

        return np.array(image)

    def attack_dataset(self, dataset):
        attacked_dataset = []
        for image in dataset:
            attacked_image = self.apply_patch(image)
            attacked_dataset.append(attacked_image)
        return np.array(attacked_dataset)
