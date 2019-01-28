from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import datetime
import os
slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'eval_dir', 'D:\\viNet\RnD\\results\\summaries\\' +
    'eval_' +
    datetime.datetime.now().strftime("%Y_%m_%d_%H_%M%S"),
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 300,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'sgd',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float(
    'opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'vinet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'validation', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', "D:\\viNet\RnD\data\stage\\tfrecord\\e3_original\\train", 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v2', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 4, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', 112, 'Eval image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 100000,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path','D:\\viNet\RnD\\results\\summaries\\Inception_2018_08_07_12_1149' ,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

def evaluate_slim(_):
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=True)
        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2,
            common_queue_min=1)
        [images,label] = provider.get(['image','label'])
        label -= FLAGS.labels_offset
        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
        #images = image_preprocessing_fn(images, eval_image_size, eval_image_size)
        images = image_preprocessing_fn(
                images, eval_image_size, eval_image_size)
        images = tf.reshape(images, [eval_image_size, eval_image_size, 3])
        images, labels = tf.train.batch(
            [images, label],
            batch_size=len(dataset.data_sources),
            num_threads=1,
            capacity=len(dataset.data_sources))

        ####################
        # Define the model #
        ####################
        logits, _ = network_fn(images)
        variables_to_restore = slim.get_variables_to_restore()

        soft = tf.nn.softmax(logits, 1)
        predictions = tf.argmax(logits, 1)
        num_batches = 1

        checkpoint_path = FLAGS.checkpoint_path
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        tf.logging.info('Evaluating %s' % checkpoint_path)

        # r_logits, r_soft, r_prediction = slim.evaluation.evaluate_once(
        #     master='',
        #     checkpoint_path=checkpoint_path,
        #     logdir=FLAGS.eval_dir,
        #     num_evals=num_batches,
        #     eval_op=[],
        #     final_op=[logits, soft, predictions],
        #     variables_to_restore=variables_to_restore)

def evaluate_tf():
    pass
    
def evaluate_keras():
    pass

if __name__ == '__main__':
    tf.app.run()
