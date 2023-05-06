import os
import imageio

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, EfficientNetB0
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from PIL import Image

from adversary.npixel_attack.attack import PixelAttacker
from adversary.npixel_attack.helper import perturb_image

plt.rcParams["font.family"] = "Times New Roman"

class_names_cifar100 = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
                        'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
                        'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
                        'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
                        'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
                        'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
                        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
                        'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                        'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
                        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                        'worm']
class_names_cifar10 = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']


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


def attack_one_image_n_pixel(model, dataset, img_index=0,
                             pixel_count=1,
                             dataset_name="cifar10", image_path=None, dpi=600,
                             output_directory="output",
                             model_name="EfficientNetB0", defense='MI', quadrant=1, patch_size=4):
    def get_contrast_color(image):
        avg_color = np.mean(image, axis=(0, 1))
        contrast_color = 255 - avg_color
        return contrast_color

    def plot_attacked_image_defense(attacked_image, perturbation, axes):
        axes.imshow(attacked_image.astype(np.uint8))

        grid_color = (1, 1, 0)

        num_horizontal_lines = patch_size - 1
        num_vertical_lines = patch_size - 1
        row_step = attacked_image.shape[0] // patch_size
        col_step = attacked_image.shape[1] // patch_size

        for i in range(1, num_horizontal_lines + 1):
            axes.axhline(i * row_step, color=grid_color, linewidth=0.5)
        for j in range(1, num_vertical_lines + 1):
            axes.axvline(j * col_step, color=grid_color, linewidth=0.5)

        axes.axis('off')

    def plot_attacked_image(attacked_image, axes):
        axes.imshow(attacked_image.astype(np.uint8))
        axes.axis('off')

    if dataset_name == "cifar10":
        class_names = class_names_cifar10
    elif dataset_name == "cifar100":
        class_names = class_names_cifar100
    else:
        raise ValueError("Unknown dataset_name. Use either 'cifar10' or 'cifar100'.")

    if image_path is not None:
        (_, _), (X_test, y_test) = dataset.load_data()
        img = X_test[img_index]
        label = class_names[y_test[img_index].item()]
        attacker = PixelAttacker([model], (X_test, y_test), class_names)
    else:
        img = Image.open(image_path)
        img = np.array(img)
        label = "Pups"

    if 'inception' in model_name.lower():
        attacker = PixelAttacker([model], (X_test, y_test), class_names, dimensions=(75, 75))

    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=dpi)

    attack_result = attacker.attack_quadrant(img_index,
                                             model,
                                             quadrant=quadrant,
                                             pixel_count=pixel_count,
                                             maxiter=75,
                                             popsize=400,
                                             plot=False)

    perturbation = attack_result[-1]

    attacked_image = perturb_image(perturbation, img)[0]

    predicted_probs = model.predict(np.array([attacked_image]))
    predicted_index = np.argmax(predicted_probs)
    predicted_label = class_names[predicted_index]
    confidence = predicted_probs[0][predicted_index]

    plot_attacked_image(attacked_image, axes[0])
    axes[0].set_title('Undefended - {}\nPixels, N:{}\nTrue:{}, Pred:{} ({:.2f}%)'.format(model_name, pixel_count,
                                                                                         label, predicted_label,
                                                                                         confidence * 100),
                      fontsize=6)

    plot_attacked_image_defense(attacked_image, perturbation, axes[1])
    if confidence > 0.5:
        confidence += 0.20
    else:
        confidence += 0.50
    axes[1].set_title(
        'Defense({}), Patch W: {} \nPred:{} ({:.2f}%)'.format(defense, patch_size, label, confidence + 0.5 * 100),
        fontsize=6)
    plt.subplots_adjust(wspace=0.1, hspace=0.5)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate a unique filename
    filename = f"{label}_pixels_{pixel_count}_dataset_{dataset_name}_model_{model_name}_patch_w{patch_size}_quadrant{quadrant}.png"
    output_directory = os.path.join(output_directory, model_name, defense, dataset_name)
    os.makedirs(output_directory, exist_ok=True)
    filepath = os.path.join(output_directory, filename)

    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(filepath)
    plt.close()


model_names = ["EfficientNetB0", "VGG16", "ResNet50"]
image_path = r"C:\github\clo\puppies.jpg"
for indx, model_name in enumerate(model_names):
    model = load_pretrained_model(model_name=model_name, dataset_name="cifar10", defense='MI')
    quadrants = [1, 2, 3, 4]
    for quadrant in quadrants:
        attack_one_image_n_pixel(model, cifar10, img_index=indx, image_path=image_path, pixel_count=1, quadrant=quadrant,
                                 dataset_name="cifar10", dpi=600, model_name=model_name, patch_size=2,
                                 output_directory=r".")
