from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

import dataset_utils

_DATA_URL = ''

# The number of images in the validation set.
_NUM_VALIDATION = 0

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir, tracks, subsample_minimum):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    dataset_root = os.path.join(dataset_dir)
    directories = []
    class_names = []
    for filename in os.listdir(dataset_root):
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    training_filenames = []
    files_per_class = 0
    subsirs = []
    minimum_number_of_files = 1e100
    if tracks:
        for directory in directories:
            subdirs = os.listdir(directory)
            for subdir in subdirs:
                path = os.path.join(dataset_root, directory, subdir)
                for filename in os.listdir(path):
                    filepath = os.path.join(path, filename)
                    training_filenames.append(filepath)
    else:
        if subsample_minimum:
            for directory in directories:
                files = os.listdir(directory)
                if len(files) < minimum_number_of_files:
                    minimum_number_of_files = len(files)
        for directory in directories:
            files = os.listdir(directory)
            print("Processing {}, number of samples:{} ".format(
                directory, len(files)))
            if subsample_minimum:
                for i in range(minimum_number_of_files):
                    filename = files[i]
                if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith(".png"):
                    print("Error: Unknown file format \'{}\'".format(filename))
                    continue
                    path = os.path.join(directory, filename)
                    training_filenames.append(path)
                    files_per_class += 1
            else:
                for filename in files:
                    if not filename.endswith('.jpg') and not filename.endswith('.jpeg') and not filename.endswith(".png"):
                        print("Error: Unknown file format \'{}\'".format(filename))
                        continue
                    path = os.path.join(directory, filename)
                    training_filenames.append(path)
                    files_per_class += 1
    print("Successfully processed {} samples from  {} sub directories".format(
        len(training_filenames), directories))
    return training_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id, prefix):
    if prefix is None:
        raise ValueError("Please supply tfrecord name prefix")
    output_filename = prefix+'_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _cc_ppor(measure_type):
    pass


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, tracks, normalize, prefix, ppor=None):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
      ppor: image preprocessing based on Patch ordering and Reconstruction 
    """
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id, prefix)
                # if not os.path.exists(output_filename):
                #       os.makedirs(output_filename)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) *
                                  num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i + 1, len(filenames), shard_id))
                            sys.stdout.flush()

                            # TODO: stick p-por here

                            # Read the filename:
                            image_data = tf.gfile.FastGFile(
                                filenames[i], 'rb').read()

                            height, width = image_reader.read_image_dims(
                                sess, image_data)

                            class_name = os.path.basename(
                                os.path.dirname(filenames[i]))

                            class_id = class_names_to_ids[class_name]

                            example = dataset_utils.image_to_tfexample(
                                image_data, b'jpg', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except:
                            print("Unable to process_loss {} file".format(
                                filenames[i]))

    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists(dataset_dir, prefix):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id, prefix)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir, output_dir, dataset_name, set_type='validation', tracks=False, subsample=False, normalize=False, prefix=None, ppor='mi'):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir, prefix):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    training_filenames, class_names = _get_filenames_and_classes(
        dataset_dir, tracks, subsample)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and test:
    random.seed(_RANDOM_SEED)
    random.shuffle(training_filenames)
    _NUM_VALIDATION = 6121

    training_filenames = training_filenames[_NUM_VALIDATION:]
    validation_filenames = training_filenames[:_NUM_VALIDATION]

    output_dir = os.path.join(output_dir, dataset_name, set_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames,
                     class_names_to_ids,
                     output_dir,
                     tracks,
                     normalize,
                     prefix)
    _convert_dataset('validation', validation_filenames,
                     class_names_to_ids,
                     output_dir,
                     False,
                     normalize,
                     prefix)

    # Finally, write the names file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, output_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\nFinished converting the {} dataset!'.format(set_type))


if __name__ == '__main__':
    DATASET_DIR = "E:\\Thesis\\data\\cifar10\\train"
    OUTPUT_DIR = 'E:\\Thesis\\data'
    NAME = "cifar10"
    TYPE = 'train'
    PREFIX = "cifar10"
    run(DATASET_DIR, OUTPUT_DIR, NAME, TYPE, False,
        subsample=False, normalize=False, prefix=PREFIX, ppor='')
