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
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
from enum import Enum

import tensorflow as tf
# from config.train_config import Config
# from configurations import TrainingConfig as Config

import logger
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from configurations import TrainingFlags
from clo import SyllabusFactory

_logger = logger.configure('tensorflow', __name__, console=False)


class TrainerType(Enum):
    slim = "Use slim.train routine to kick-off training"
    tf = "Use explicit tensorflow session "


class Trainer(object):
    _config = None
    _slim = None
    _train_dir = None
    _measure_index = 0
    _measure_list = None
    _measure = None

    def __init__(self, config: TrainingFlags):
        self._config = config
        self._slim = tf.contrib.slim

        assert self._config.measure is None or self._config.measure_list is None, \
            "Must supply either a measure or a list of measures "
        if self._config.measure:
            self._measure = self._config.measure
        elif self._config.measure_list:
            _logger.info("Selecting ranking measure at index {} from list, measures: "
                         "{}".format(self._measure_index, self._config.measure_list))

            self._measure_list = self._config.measure_list.split(',')
            self._measure = self._measure_list[self._measure_index]

    @tf.function
    def _propose_syllabus(self, graph, images, labels):
        assert graph
        sf = SyllabusFactory(graph, images, labels, self._config.batch_size)
        images, labels = sf.propose_syllabus(self._measure, self._config.ordering)

        return images, labels

    def _write_config(self):
        """
        Setup logging directory and write training configuration to that directory
        """
        if not os.path.exists(self._train_dir):
            os.makedirs(self._train_dir)

        file = os.path.join(self._train_dir, self._config.model_name + "_" + self._config.dataset_name + ".yaml")
        if os.path.exists(file):
            _logger.warn("Training dir already exists, overriding")
        self._config.dump(file)
        _logger.debug("Successfully saved configuration file to train_dir, file: {}".format(file))

    def _configure_learning_rate(self, num_samples_per_epoch, global_step):
        """Configures the learning rate.

        Args:
          num_samples_per_epoch: The number of samples in each epoch of training.
          global_step: The global_step tensor.

        Returns:
          A `Tensor` representing the learning rate.

        Raises:
          ValueError: if
        """
        decay_steps = int(num_samples_per_epoch / self._config.batch_size *
                          self._config.num_epochs_per_decay)
        if self._config.sync_replicas:
            decay_steps /= self._config.replicas_to_aggregate

        if self._config.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(self._config.learning_rate,
                                              global_step,
                                              decay_steps,
                                              self._config.learning_rate_decay_factor,
                                              staircase=True,
                                              name='exponential_decay_learning_rate')
        elif self._config.learning_rate_decay_type == 'fixed':
            return tf.constant(self._config.learning_rate, name='fixed_learning_rate')
        elif self._config.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(self._config.learning_rate,
                                             global_step,
                                             decay_steps,
                                             self._config.end_learning_rate,
                                             power=1.0,
                                             cycle=False,
                                             name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized',
                             self._config.learning_rate_decay_type)

    def _configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.

        Raises:
          ValueError: if config.OPTIMIZER is not recognized.
          :param learning_rate: A scalar or `Tensor` learning rate.
          :return: An instance of an optimizer.
        """
        if self._config.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=self._config.adadelta_rho,
                epsilon=self._config.opt_epsilon)
        elif self._config.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=self._config.adagrad_initial_accumulator_value)
        elif self._config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=self._config.adam_beta1,
                beta2=self._config.adam_beta2,
                epsilon=self._config.opt_epsilon)
        elif self._config.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=self._config.ftrl_learning_rate_power,
                initial_accumulator_value=self._config.ftrl_initial_accumulator_value,
                l1_regularization_strength=self._config.ftrl_l1,
                l2_regularization_strength=self._config.ftrl_l2)
        elif self._config.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=self._config.momentum,
                name='Momentum')
        elif self._config.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=self._config.rmsprop_decay,
                momentum=self._config.rmsprop_momentum,
                epsilon=self._config.opt_epsilon)
        elif self._config.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self._config.optimizer)
        return optimizer

    def _get_init_fn(self):
        """Returns a function time_dist by the chief worker to warm-start the training.

        Note that the init_fn is only time_dist when initializing the model during the very
        first global step.

        Returns:
          An init function time_dist by the supervisor.
        """
        # print ("CHECKPOINT_PATH = " + config.CHECKPOINT_PATH)
        if self._config.checkpoint_path is None:
            return None

        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
        if tf.train.latest_checkpoint(self._train_dir):
            tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % self._train_dir)
            return None

        exclusions = []
        if self._config.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in self._config.checkpoint_exclude_scopes.split(',')]

        variables_to_restore = []
        for var in self._slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        if tf.gfile.IsDirectory(self._config.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self._config.checkpoint_path)
        else:
            checkpoint_path = self._config.checkpoint_path

        tf.logging.info('Fine-tuning from %s' % checkpoint_path)

        return self._slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=self._config.ignore_missing_vars)

    def _get_variables_to_train(self):
        """Returns a list of variables to train.

        Returns:
          A list of variables to train by the optimizer.
        """
        if self._config.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self._config.trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def _train_slim(self):
        if not self._config.dataset_dir:
            raise ValueError(
                'You must supply the dataset directory with --dataset_dir')

        tf.logging.set_verbosity(tf.logging.DEBUG)

        # syllabus optimization output directory
        if not self._config.measure_list:
            self._train_dir = os.path.join(self._config.training_log_dir,
                                           self._config.dataset_name,
                                           self._config.model_name,
                                           self._measure,
                                           str(self._config.max_number_of_steps))
        else:
            measure_list = self._config.measure_list.replace(",", "_")
            self._train_dir = os.path.join(self._config.training_log_dir,
                                           self._config.dataset_name,
                                           self._config.model_name,
                                           measure_list,
                                           str(self._config.max_number_of_steps))

        self._write_config()
        my_g = tf.Graph()
        with my_g.as_default():
            #######################
            # Config model_deploy #
            #######################
            deploy_config = model_deploy.DeploymentConfig(
                num_clones=self._config.num_clones,
                clone_on_cpu=self._config.clone_on_cpu,
                replica_id=self._config.task,
                num_replicas=self._config.worker_replicas,
                num_ps_tasks=self._config.num_ps_tasks)

            # Create global_step
            with tf.device(deploy_config.variables_device()):
                global_step = tf.train.create_global_step()

            ######################
            # Select the dataset #
            ######################
            dataset = dataset_factory.get_dataset(
                self._config.dataset_name, self._config.dataset_split_name,
                self._config.dataset_dir)

            ######################
            # Select the network #
            ######################
            network_fn = nets_factory.get_network_fn(
                self._config.model_name,
                num_classes=(dataset.num_classes - self._config.labels_offset),
                weight_decay=self._config.weight_decay,
                is_training=True)

            #####################################
            # Select the preprocessing function #
            #####################################
            preprocessing_name = self._config.preprocessing_name or self._config.model_name
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=True)

            syllabus_proposed = False
            ##############################################################
            # Create a dataset provider that loads data from the dataset #
            ##############################################################
            with tf.device(deploy_config.inputs_device()):
                provider = self._slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=self._config.num_readers,
                    common_queue_capacity=20 * self._config.batch_size,
                    common_queue_min=10 * self._config.batch_size)
                [image, label] = provider.get(['image', 'label'])

                label -= self._config.labels_offset

                train_image_size = self._config.train_image_size or network_fn.default_image_size
                image = image_preprocessing_fn(
                    image, train_image_size, train_image_size, self._measure,
                    self._config.ordering, self._config.patch_size)

                images, labels = tf.train.batch(
                    [image, label],
                    batch_size=self._config.batch_size,
                    num_threads=self._config.num_preprocessing_threads,
                    capacity=5 * self._config.batch_size,
                    allow_smaller_final_batch=True)

                # images, labels = tf.data.Dataset.batch(self._config.batch_size)

                # Setup syllabus learning optimization #

                sf = None
                if self._config.curriculum:
                    if self._config.measure_list:
                        self._measure_index += 1
                        self._measure = self._measure_list[self._measure_index]
                        tf.logging.info("Updated syllabus, ranking measure: {}".format(self._measure))
                    # sf = SyllabusFactory(images, labels, self._config.batch_size)

                    images, labels = self._propose_syllabus(my_g, images, labels)
                    syllabus_proposed = True

                # syllabus learning

                labels = self._slim.one_hot_encoding(
                    labels, dataset.num_classes - self._config.labels_offset)

                _batch_queue = self._slim.prefetch_queue.prefetch_queue(
                    [images, labels], capacity=2 * deploy_config.num_clones)

            ####################
            # Define the model #
            ####################
            def clone_fn(batch_queue):
                """Allows data parallelism by creating multiple clones of network_fn."""
                images, labels = batch_queue.dequeue()

                if self._config.curriculum:
                    if self._config.measure_list:
                        self._measure_index += 1
                        self._measure = self._measure_list[self._measure_index]
                        tf.logging.info("Updated syllabus, ranking measure: {}".format(self._measure))
                    # sf = SyllabusFactory(images, labels, self._config.batch_size)
                    images_curriculum, labels_curriculum = self._propose_syllabus(images, labels)
                    logits, end_points = network_fn(images_curriculum)
                else:
                    logits, end_points = network_fn(images)

                #############################
                # Specify the loss function #
                #############################
                if 'AuxLogits' in end_points:
                    self._slim.losses.softmax_cross_entropy(
                        end_points['AuxLogits'], labels,
                        label_smoothing=self._config.label_smoothing, weights=0.4,
                        scope='aux_loss')
                self._slim.losses.softmax_cross_entropy(
                    logits, labels, label_smoothing=self._config.label_smoothing, weights=1.0)
                return end_points

            # Gather initial summaries.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            clones = model_deploy.create_clones(
                deploy_config, clone_fn, [_batch_queue])
            first_clone_scope = deploy_config.clone_scope(0)
            # Gather update_ops from the first clone. These contain, for example,
            # the updates for the batch_norm variables created by network_fn.
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, first_clone_scope)

            # Add summaries for end_points.
            end_points = clones[0].outputs
            for end_point in end_points:
                x = end_points[end_point]
                summaries.add(tf.summary.histogram('activations/' + end_point, x))
                summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                                tf.nn.zero_fraction(x)))
            # Add summaries for losses.
            for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
                summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

            # Add summaries for variables.
            for variable in self._slim.get_model_variables():
                summaries.add(tf.summary.histogram(variable.op.name, variable))

            #################################
            # Configure the moving averages #
            #################################
            if self._config.moving_average_decay:
                moving_average_variables = self._slim.get_model_variables()
                variable_averages = tf.train.ExponentialMovingAverage(
                    self._config.moving_average_decay, global_step)
            else:
                moving_average_variables, variable_averages = None, None

            #########################################
            # Configure the optimization procedure. #
            #########################################
            with tf.device(deploy_config.optimizer_device()):
                learning_rate = self._configure_learning_rate(
                    dataset.num_samples, global_step)
                optimizer = self._configure_optimizer(learning_rate)
                summaries.add(tf.summary.scalar('learning_rate', learning_rate))

            if self._config.sync_replicas:
                # If sync_replicas is enabled, the averaging will be done in the chief
                # queue runner.
                optimizer = tf.train.SyncReplicasOptimizer(
                    opt=optimizer,
                    replicas_to_aggregate=self._config.replicas_to_aggregate,
                    total_num_replicas=self._config.worker_replicas,
                    variable_averages=variable_averages,
                    variables_to_average=moving_average_variables)
            elif self._config.moving_average_decay:
                # Update ops executed locally by trainer.
                update_ops.append(variable_averages.apply(
                    moving_average_variables))

            # Variables to train.
            variables_to_train = self._get_variables_to_train()

            #  and returns a train_tensor and summary_op
            total_loss, clones_gradients = model_deploy.optimize_clones(
                clones,
                optimizer,
                var_list=variables_to_train)

            # Add total_loss to summary.
            summaries.add(tf.summary.scalar('total_loss', total_loss))
            summaries.add(tf.summary.image("input/images", images))
            # Create gradient updates.
            grad_updates = optimizer.apply_gradients(clones_gradients, global_step=global_step)
            update_ops.append(grad_updates)

            update_op = tf.group(*update_ops)
            with tf.control_dependencies([update_op]):
                train_tensor = tf.identity(total_loss, name='train_op')

            # Add the summaries from the first clone. These contain the summaries
            # created by model_fn and either optimize_clones() or _gather_clone_loss().
            summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                               first_clone_scope))

            # Merge all summaries together.
            summary_op = tf.summary.merge(list(summaries), name='summary_op')
            session_config = tf.ConfigProto(allow_soft_placement=False)

            ##########################
            # Kicks off the training. #
            ###########################
            self._slim.learning.train(
                    train_tensor,
                    logdir=self._train_dir,
                    master=self._config.master,
                    is_chief=(self._config.task == 0),
                    init_fn=self._get_init_fn(),
                    summary_op=summary_op,
                    number_of_steps=self._config.max_number_of_steps,
                    log_every_n_steps=self._config.log_every_n_steps,
                    save_summaries_secs=self._config.save_summaries_secs,
                    save_interval_secs=self._config.save_interval_secs,
                    sync_optimizer=optimizer if self._config.sync_replicas else None,
                    session_config=session_config)

    def _train_tf(self):
        if not self._config.dataset_dir:
            raise ValueError(
                'You must supply the dataset directory with --dataset_dir')

        tf.logging.set_verbosity(tf.logging.DEBUG)

        # syllabus optimization output directory
        if not self._config.measure_list:
            self._train_dir = os.path.join(self._config.training_log_dir,
                                           self._config.dataset_name,
                                           self._config.model_name,
                                           self._measure,
                                           str(self._config.max_number_of_steps))
        else:
            measure_list = self._config.measure_list.replace(",", "_")
            self._train_dir = os.path.join(self._config.training_log_dir,
                                           self._config.dataset_name,
                                           self._config.model_name,
                                           measure_list,
                                           str(self._config.max_number_of_steps))

        self._write_config()

    def run(self, trainer_type: TrainerType):
        if trainer_type == TrainerType.slim:
            return self._train_slim()
        else:
            return self._train_tf()
