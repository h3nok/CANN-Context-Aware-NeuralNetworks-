import os
import shutil
import pathlib

tiny_imagenet_path = 'tiny-imagenet-200'
output_path = 'tiny-imagenet-1k'

# Create output directories
os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)

# Convert training data
for class_folder in os.listdir(os.path.join(tiny_imagenet_path, 'train')):
    src_folder = os.path.join(tiny_imagenet_path, 'train', class_folder, 'images')
    dst_folder = os.path.join(output_path, 'train', class_folder)
    shutil.copytree(src_folder, dst_folder)

# Convert validation data
val_annotations_file = os.path.join(tiny_imagenet_path, 'val', 'val_annotations.txt')
val_annotations = {}

# Read validation annotations
with open(val_annotations_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        image_name, class_id = parts[0], parts[1]
        val_annotations[image_name] = class_id

# Copy validation images to the corresponding class folders
val_images_folder = os.path.join(tiny_imagenet_path, 'val', 'images')
for image_name, class_id in val_annotations.items():
    src_file = os.path.join(val_images_folder, image_name)
    dst_folder = os.path.join(output_path, 'val', class_id)
    os.makedirs(dst_folder, exist_ok=True)
    dst_file = os.path.join(dst_folder, image_name)
    shutil.copy(src_file, dst_file)
