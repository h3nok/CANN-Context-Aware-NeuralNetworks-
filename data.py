import sys
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
from util import *


def create_label(image_name):
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1, 0])
    if word_label == 'dog':
        return np.array([0, 1])


def create_train_data(numpy_file_name, train_set_dir, img_size=32):
    print("Entering create_train_data, npy output file name = {}".format(numpy_file_name))
    output_file = numpy_file_name
    if not os.path.exists(train_set_dir):
        raise ValueError(
            "Dataset directory  \"{}\" not found.".format(train_set_dir))
        return

    prompt = True
    if os.path.exists(output_file):
        prompt = query_yes_no(
            "Output file name \"{}\" already exists. Do you want to override it?".format(output_file))
    if prompt:
        print("Creating training data, {} ...".format(numpy_file_name))
        training_data = []
        for img in tqdm(os.listdir(train_set_dir)):
            path = os.path.join(train_set_dir, img)
            img_data = cv2.imread(path, cv2.IMREAD_COLOR)
            try:
                img_data = cv2.resize(img_data, (img_size, img_size))
            except:
                print("Unable to resize")
                continue
            training_data.append([np.array(img_data), create_label(img)])

        print ("Saving {} to disk".format(output_file))
        np.save(output_file, training_data)

        return training_data


def create_test_data(numpy_file_name, test_set_dir, img_size=32):
    print("Entering create_test_data, npy output file name = {}".format(numpy_file_name))
    output_file = numpy_file_name
    prompt = True

    if not os.path.exists(test_set_dir):
        raise ValueError(
            "Dataset directory  \"{}\" not found.".format(test_set_dir))

    if os.path.exists(output_file):
        prompt = query_yes_no(
            "Output file name \"{}\" already exists. Do you want to override it?".format(output_file))

    if prompt:
        print("Creating testing data ...")
        testing_data = []
        for img in tqdm(os.listdir(test_set_dir)):
            path = os.path.join(test_set_dir, img)
            img_num = img.split('.')[0]
            img_data = cv2.imread(path, cv2.IMREAD_COLOR)
            img_data = cv2.resize(img_data, (img_size, img_size))
            testing_data.append([np.array(img_data), img_num])
        print ("Saving {} to disk".format(output_file))
        # shuffle(testing_data)
        np.save(output_file, testing_data)

        return testing_data
