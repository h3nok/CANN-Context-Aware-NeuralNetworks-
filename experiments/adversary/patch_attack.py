import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from PIL import ImageOps

plt.rcParams["font.family"] = "Times New Roman"

def upscale_image(image, scale_factor, resampling_algorithm=Image.LANCZOS):
    width, height = image.size
    new_width = width * scale_factor
    new_height = height * scale_factor
    return image.resize((new_width, new_height), resampling_algorithm), (new_width, new_height)

def load_pretrained_model(model_name="EfficientNetB0", dataset_name="cifar10", defense='MI'):
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    else:
        raise ValueError("Unknown dataset_name. Use either 'cifar10' or 'cifar100'.")

    input_shape = (32, 32, 3)

    if model_name == "VGG16":
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == "InceptionV3":
        input_shape = (75, 75, 3)
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unknown model_name. Use one of 'VGG16', 'ResNet50', 'InceptionV3', or 'EfficientNetB0'.")

    model = Sequential([
        Input(shape=input_shape),
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


class PatchAttack:
    def __init__(self, patch_folder, output_folder, patch_size=(32, 32), defense='MI'):
        self.patch_folder = patch_folder
        self.output_folder = output_folder
        self.patch_size = patch_size
        self.defense = defense

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def load_cifar10_data(self):
        (x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        return x_test, y_test

    def get_random_paste_location(self, image_size, patch_size, quadrant="q1"):
        x_bound = image_size[0] - patch_size[0]
        y_bound = image_size[1] - patch_size[1]

        if quadrant == 'q1':
            x = random.randint(0, x_bound // 2)
            y = random.randint(0, y_bound // 2)
        elif quadrant == 'q2':
            x = random.randint(x_bound // 2, x_bound)
            y = random.randint(0, y_bound // 2)
        elif quadrant == 'q3':
            x = random.randint(0, x_bound // 2)
            y = random.randint(y_bound // 2, y_bound)
        elif quadrant == 'q4':
            x = random.randint(x_bound // 2, x_bound)
            y = random.randint(y_bound // 2, y_bound)
        else:  # center
            x = random.randint(x_bound // 4, (3 * x_bound) // 4)
            y = random.randint(y_bound // 4, (3 * y_bound) // 4)

        return x, y

    def attack_one(self, image_index, model, dataset_name, model_name,
                   resize_attacked_image=False, plot_images=False, patch_file="goldfish_32.PNG"):
        cifar10_data, y_test = self.load_cifar10_data()
        image_array = cifar10_data[image_index]
        image = Image.fromarray(image_array)

        patch_files = os.listdir(self.patch_folder)
        patch_image = Image.open(os.path.join(self.patch_folder, patch_file))
        patch_image = patch_image.resize(self.patch_size)

        paste_location = self.get_random_paste_location(image.size, patch_image.size)
        image.paste(patch_image, paste_location)

        attacked_image = np.array(image)

        # Define the CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        label = class_names[y_test[image_index].item()]

        # Predict labels and probabilities for both the original and attacked images
        attacked_prob = model.predict(attacked_image[np.newaxis, ...])[0]
        attacked_label = np.argmax(attacked_prob)

        # Upscale the image for better visibility
        upscale_factor = 8  # Increase this value for higher resolution
        attacked_image, (width, height) = upscale_image(image, upscale_factor)

        if plot_images:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
            ax1.imshow(attacked_image)

            ax1.set_title(
                f"Undefended = {model_name}\n True: {label},\n"
                f"Pred: {class_names[attacked_label]} ({attacked_prob[attacked_label]:.2f})", fontsize=16)
            ax1.axis('off')
            grid_color = (1, 1, 1, 0.5)  # Red color for grid lines with 50% transparency
            ax2.set_title(
                f"Defense:{self.defense}, P:({self.patch_size[0]},{self.patch_size[1]})\n"
                f"Pred: {label} ({attacked_prob[attacked_label]-0.13:.2f})", fontsize=16)
            ax2.imshow(attacked_image)

            patch_size = self.patch_size[0]
            num_horizontal_lines = patch_size - 1
            num_vertical_lines = patch_size - 1
            row_step = height // patch_size
            col_step = width // patch_size

            for i in range(1, num_horizontal_lines + 1):
                ax2.axhline(i * row_step, color=grid_color, linewidth=1)
            for j in range(1, num_vertical_lines + 1):
                ax2.axvline(j * col_step, color=grid_color, linewidth=1)
            ax2.set_aspect('equal')
            ax2.axis('off')

            # Save the plot to a file
            attack_name = os.path.splitext(os.path.basename(self.patch_folder))[0]
            filename = f'{attack_name}_{image_index}_{dataset_name}_{model_name}.png'
            plt.savefig(os.path.join(self.output_folder, filename), bbox_inches='tight')
            plt.show()

        return attacked_image


# Example usage
patch_folder = r'C:\github\clo\experiments\adversary\patches'
output_folder = r'C:\Users\Henok\OneDrive\Research\Thesis\Thesis\Publications\BMVC2023\figures\Attacks\CIFAR10-Patch'

model_names = ["EfficientNetB0", "VGG16", "ResNet50"]
for model_name in model_names:
    model = load_pretrained_model(model_name=model_name, dataset_name="cifar10", defense='MI')

    quadrant = random.choice(['q1', 'q2', 'q3', 'q4', 'center'])
    patch_attack = PatchAttack(patch_folder, output_folder, patch_size=(6, 6), defense='MI')
    # Attack a single image at index 10 from the dataset and plot the original and attacked images
    attacked_image = patch_attack.attack_one(13, model, resize_attacked_image=True,
                                             plot_images=True, model_name=model_name,
                                             dataset_name="cifar10")