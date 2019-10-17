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

import tensorflow as tf
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
# from config.train_config import Config
from configurations import TrainingConfig as Config
import logger

_logger = logger.configure(__name__, console=True)


class Trainer(object):
    config = None
    slim = None

    def __init__(self, config: Config):
        self.config = config
        self.slim = tf.contrib.slim

    def _write_out_config(self):
        self.config.params.train_dir = os.path.join(self.config.params.train_dir, self.config.params.customer,
                                                    self.config.params.model_name +
                                                    "_" + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M%S"))
        if not os.path.exists(self.config.params.train_dir):
            os.makedirs(self.config.params.train_dir)

        file = os.path.join(self.config.params.train_dir, self.config.params.model_name + ".yaml")
        self.config.dump_yaml(file)

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
        decay_steps = int(num_samples_per_epoch / self.config.params.batch_size *
                          self.config.params.num_epochs_per_decay)
        if self.config.params.sync_replicas:
            decay_steps /= self.config.params.replicas_to_aggregate

        if self.config.params.learning_rate_decay_type == 'exponential':
            return tf.train.exponential_decay(self.config.params.learning_rate,
                                              global_step,
                                              decay_steps,
                                              self.config.params.learning_rate_decay_factor,
                                              staircase=True,
                                              name='exponential_decay_learning_rate')
        elif self.config.params.learning_rate_decay_type == 'fixed':
            return tf.constant(self.config.params.learning_rate, name='fixed_learning_rate')
        elif self.config.params.learning_rate_decay_type == 'polynomial':
            return tf.train.polynomial_decay(self.config.params.learning_rate,
                                             global_step,
                                             decay_steps,
                                             self.config.params.end_learning_rate,
                                             power=1.0,
                                             cycle=False,
                                             name='polynomial_decay_learning_rate')
        else:
            raise ValueError('learning_rate_decay_type [%s] was not recognized',
                             self.config.params.learning_rate_decay_type)

    def _configure_optimizer(self, learning_rate):
        """Configures the optimizer used for training.

        Raises:
          ValueError: if config.OPTIMIZER is not recognized.
          :param learning_rate: A scalar or `Tensor` learning rate.
          :return: An instance of an optimizer.
        """
        if self.config.params.optimizer == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate,
                rho=self.config.params.adadelta_rho,
                epsilon=self.config.params.opt_epsilon)
        elif self.config.params.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(
                learning_rate,
                initial_accumulator_value=self.config.params.adagrad_initial_accumulator_value)
        elif self.config.params.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=self.config.params.adam_beta1,
                beta2=self.config.params.adam_beta2,
                epsilon=self.config.params.opt_epsilon)
        elif self.config.params.optimizer == 'ftrl':
            optimizer = tf.train.FtrlOptimizer(
                learning_rate,
                learning_rate_power=self.config.params.ftrl_learning_rate_power,
                initial_accumulator_value=self.config.params.ftrl_initial_accumulator_value,
                l1_regularization_strength=self.config.params.ftrl_l1,
                l2_regularization_strength=self.config.params.ftrl_l2)
        elif self.config.params.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                learning_rate,
                momentum=self.config.params.momentum,
                name='Momentum')
        elif self.config.params.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                decay=self.config.params.rmsprop_decay,
                momentum=self.config.params.rmsprop_momentum,
                epsilon=self.config.params.opt_epsilon)
        elif self.config.params.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Optimizer [%s] was not recognized', self.config.params.optimizer)
        return optimizer

    def _get_init_fn(self):
        """Returns a function time_dist by the chief worker to warm-start the training.

        Note that the init_fn is only time_dist when initializing the model during the very
        first global step.

        Returns:
          An init function time_dist by the supervisor.
        """
        # print ("CHECKPOINT_PATH = " + config.CHECKPOINT_PATH)
        if self.config.params.checkpoint_path is None:
            return None

        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
        if tf.train.latest_checkpoint(self.config.params.train_dir):
            tf.logging.info(
                'Ignoring --checkpoint_path because a checkpoint already exists in %s'
                % self.config.params.train_dir)
            return None

        exclusions = []
        if self.config.params.checkpoint_exclude_scopes:
            exclusions = [scope.strip()
                          for scope in self.config.params.checkpoint_exclude_scopes.split(',')]

        variables_to_restore = []
        for var in self.slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)

        if tf.gfile.IsDirectory(self.config.params.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(self.config.params.checkpoint_path)
        else:
            checkpoint_path = self.config.params.checkpoint_path

        tf.logging.info('Fine-tuning from %s' % checkpoint_path)

        return self.slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=self.config.params.ignore_missing_vars)

    def _get_variables_to_train(self):
        """Returns a list of variables to train.

        Returns:
          A list of variables to train by the optimizer.
        """
        if self.config.params.trainable_scopes is None:
            return tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in self.config.params.trainable_scopes.split(',')]

        variables_to_train = []
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)
        return variables_to_train

    def train(self):
        if not self.config.params.train_dir:
            raise ValueError(
                'You must supply the dataset directory with --dataset_dir')

        tf.logging.set_verbosity(tf.logging.INFO)
        self._write_out_config()
        with tf.Graph().as_default():
            #######################
            # Config model_deploy #
            #######################
            deploy_config = model_deploy.DeploymentConfig(
                num_clones=self.config.params.num_clones,
                clone_on_cpu=self.config.params.clone_on_cpu,
                replica_id=self.config.params.task,
                num_replicas=self.config.params.worker_replicas,
                num_ps_tasks=self.config.params.num_ps_tasks)

            # Create global_step
            with tf.device(deploy_config.variables_device()):
                global_step = tf.train.create_global_step()

            ######################
            # Select the dataset #
            ######################
            dataset = dataset_factory.get_dataset(
                self.config.params.dataset_name, self.config.params.dataset_split_name, self.config.params.dataset_dir)

            ######################
            # Select the network #
            ######################
            network_fn = nets_factory.get_network_fn(
                self.config.params.model_name,
                num_classes=(dataset.num_classes - self.config.params.labels_offset),
                weight_decay=self.config.params.weight_decay,
                is_training=True)

            #####################################
            # Select the preprocessing function #
            #####################################
            preprocessing_name = self.config.params.preprocessing_name or self.config.params.model_name
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name,
                is_training=True)

            ##############################################################
            # Create a dataset provider that loads data from the dataset #
            ##############################################################
            with tf.device(deploy_config.inputs_device()):
                provider = self.slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=self.config.params.num_readers,
                    common_queue_capacity=20 * self.config.params.batch_size,
                    common_queue_min=10 * self.config.params.batch_size)
                [image, label] = provider.get(['image', 'label'])
                label -= self.config.params.labels_offset

                train_image_size = self.config.params.train_image_size or network_fn.default_image_size

                image = image_preprocessing_fn(
                    image, train_image_size, train_image_size)

                images, labels = tf.train.batch(
                    [image, label],
                    batch_size=self.config.params.batch_size,
                    num_threads=self.config.params.num_preprocessing_threads,
                    capacity=5 * self.config.params.batch_size)
                labels = self.slim.one_hot_encoding(
                    labels, dataset.num_classes - self.config.params.labels_offset)
                batch_queue = self.slim.prefetch_queue.prefetch_queue(
                    [images, labels], capacity=2 * deploy_config.num_clones)

            ####################
            # Define the model #
            ####################
            def clone_fn(batch_queue):
                """Allows data parallelism by creating multiple clones of network_fn."""
                images, labels = batch_queue.dequeue()
                logits, end_points = network_fn(images)

                #############################
                # Specify the loss function #
                #############################
                if 'AuxLogits' in end_points:
                    self.slim.losses.softmax_cross_entropy(
                        end_points['AuxLogits'], labels,
                        label_smoothing=self.config.params.label_smoothing, weights=0.4,
                        scope='aux_loss')
                self.slim.losses.softmax_cross_entropy(
                    logits, labels, label_smoothing=self.config.params.label_smoothing, weights=1.0)
                return end_points

            # Gather initial summaries.
            summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

            clones = model_deploy.create_clones(
                deploy_config, clone_fn, [batch_queue])
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
            for variable in self.slim.get_model_variables():
                summaries.add(tf.summary.histogram(variable.op.name, variable))

            #################################
            # configure the moving averages #
            #################################
            if self.config.params.moving_average_decay:
                moving_average_variables = self.slim.get_model_variables()
                variable_averages = tf.train.ExponentialMovingAverage(
                    self.config.params.moving_average_decay, global_step)
            else:
                moving_average_variables, variable_averages = None, None

            #########################################
            # configure the optimization procedure. #
            #########################################
            with tf.device(deploy_config.optimizer_device()):
                learning_rate = self._configure_learning_rate(
                    dataset.num_samples, global_step)
                optimizer = self._configure_optimizer(learning_rate)
                summaries.add(tf.summary.scalar('learning_rate', learning_rate))

            if self.config.params.sync_replicas:
                # If sync_replicas is enabled, the averaging will be done in the chief
                # queue runner.
                optimizer = tf.train.SyncReplicasOptimizer(
                    opt=optimizer,
                    replicas_to_aggregate=self.config.params.replicas_to_aggregate,
                    total_num_replicas=self.config.params.worker_replicas,
                    variable_averages=variable_averages,
                    variables_to_average=moving_average_variables)
            elif self.config.params.moving_average_decay:
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

            # Create gradient updates.
            grad_updates = optimizer.apply_gradients(clones_gradients,
                                                     global_step=global_step)
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

            ##########################
            # Kicks off the training. #
            ###########################
            self.slim.learning.train(
                train_tensor,
                logdir=self.config.params.train_dir,
                master=self.config.params.tf_master,
                is_chief=(self.config.params.task == 0),
                init_fn=self._get_init_fn(),
                summary_op=summary_op,
                number_of_steps=self.config.params.max_number_of_steps,
                log_every_n_steps=self.config.params.log_every_n_steps,
                save_summaries_secs=self.config.params.save_summaries_secs,
                save_interval_secs=self.config.params.save_interval_secs,
                sync_optimizer=optimizer if self.config.params.sync_replicas else None)

    def run(self):
        return self.train()


def run_trainer():
    # The config object should be created separately so that adjustments can be made before
    # giving it to the trainer
    config = Config(None)
    config.load_config_from_xml('C:\\viNet_RnD\\configs\\E3.xml')
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    run_trainer()
