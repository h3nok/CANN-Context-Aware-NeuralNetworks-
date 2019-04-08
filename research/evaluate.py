# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import glob
import math
import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from pprint import pprint
from preprocessing import preprocessing_factory

_CKPT_PATTER = "model.ckpt-"

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string('training_mode', 'curriculum', "training strategies employed to grow network")

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/deeplearning/train_log',
    #'checkpoint_path', "E:\\Thesis\\CC_V2\\summaries",
    "The directory where the model was written to or an absolute path to a "
    "checkpoint file.")
tf.app.flags.DEFINE_string('metric', 'ce', 'Metric used to generate the checkpoints')
tf.app.flags.DEFINE_integer('iter', 10000,
                            'The number of training iterations used to generate the checkpoints')

tf.app.flags.DEFINE_string(
    #'eval_dir', 'E:\\Datasets\\cifar\\cifar10\\eval_log',
    'eval_dir', '/home/deeplearning/eval',
    'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/deeplearning/data/cifar10-val',
    #'dataset_dir', 'E:\\Datasets\\cifar\cifar10\\tfrecord\\test',
    "The directory where the dataset files are stored.")

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 224, 'Eval image size')

tf.app.flags.DEFINE_string("eval", "once", "Evaluation loop (once, loop)")

FLAGS = tf.app.flags.FLAGS


def get_checkpoints():
    if not os.path.exists(FLAGS.checkpoint_path):
        raise RuntimeError("Checkpoint path not found\n")

    dir =  os.path.join(FLAGS.checkpoint_path, FLAGS.training_mode,
                                         FLAGS.model_name,
                                         FLAGS.dataset_name,
                                         FLAGS.metric,
                                         str(FLAGS.iter))
    if not os.path.exists(dir):
        raise RuntimeError("Check point direction \'{}\' not found".format(dir))
    files = glob.glob(dir+"/*")

    print("Processing {} files from {}".format(len(files), dir))
    pprint(dir)

    regex = "\w+.\w+-\d+"
    # files = os.listdir(dir)
    chkpts = set()
    for file in files:
        if _CKPT_PATTER in file:
            model_ckpt = re.search(regex, file)
            chkpts.add(os.path.join(dir,model_ckpt.group(0)))

    return chkpts


def eval(checkpoint):
    if not FLAGS.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

    if FLAGS.eval_dir:
        FLAGS.eval_dir = os.path.join(FLAGS.eval_dir,
                                      FLAGS.training_mode, FLAGS.model_name,
                                      FLAGS.metric,
                                      str(FLAGS.iter))
    else:
        raise RuntimeError("Unable to run evaluation. Summary dir not found")

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * FLAGS.batch_size,
            common_queue_min=FLAGS.batch_size)
        [image, label] = provider.get(['image', 'label'])
        label -= FLAGS.labels_offset


        #####################################
        # Select the pre-processing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

        images, labels = tf.train.batch(
            [image, label],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            "mse": slim.metrics.streaming_mean_squared_error(predictions, labels),
            # "tp": slim.metrics.(predictions, predictions),
            # "tn": tf.metrics.true_negatives(labels, predictions),
            # 'conf': tf.metrics.auc(labels, predictions)
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(
                dataset.num_samples / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint)
        else:
            checkpoint_path = checkpoint

        tf.logging.info('Evaluating %s' % checkpoint_path)

        if FLAGS.eval == 'once':
            slim.evaluation.evaluate_once(
                master=FLAGS.master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=variables_to_restore)
        else:
            slim.evaluation.evaluation_loop(
                master=FLAGS.master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=variables_to_restore)


def main(_):
    dir = "E:\\viNet_RnD\\Deployment\\E3\\inception_v2_2019_04_02_07_3822"
    ckpts = get_checkpoints()
    for ckpt in ckpts:
        eval(checkpoint=ckpt)


if __name__ == '__main__':
    tf.app.run()
