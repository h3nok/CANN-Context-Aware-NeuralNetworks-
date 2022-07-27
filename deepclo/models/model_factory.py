import os
import time
from datetime import date

import netron
import tensorflow as tf
import tqdm
from classification_models.tfkeras import Classifiers
from keras.backend import sigmoid
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
from vit_keras import vit
import numpy as np
import keras
from tensorflow.keras.utils import get_custom_objects
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from deepclo.algorithms.curriculum import Curriculum
from deepclo.algorithms.por import POR
from deepclo.config import Config
from deepclo.utils import configure_logger

keras_apps = tf.keras.applications

Models = {
    "Keras": {
        # EfficientNet Models
        'b0': keras_apps.EfficientNetB0,
        'b1': keras_apps.EfficientNetB1,
        'b2': keras_apps.EfficientNetB2,
        'b3': keras_apps.EfficientNetB3,
        'b4': keras_apps.EfficientNetB4,
        'b5': keras_apps.EfficientNetB5,
        'b6': keras_apps.EfficientNetB6,
        'b7': keras_apps.EfficientNetB7,
        'l2': keras_apps.EfficientNetV2L,
        'resnet50': keras_apps.ResNet50,
        'resnet101': keras_apps.ResNet101,
        'inception': keras_apps.InceptionV3,
        'mobilenet': keras_apps.MobileNet,
        'densenet': keras_apps.DenseNet121,
        'nasnet': keras_apps.NASNetLarge,
        'nasnetmobile': keras_apps.NASNetMobile,
        'resnext50': Classifiers.get('resnext50')[0],
        'resnext101': Classifiers.get('resnext101')[0]
    },

    # ViT-L
    "ViT-L": {
        'b16': vit.vit_b16,
        'b32': vit.vit_b32,
        'l16': vit.vit_l16,
        'l32': vit.vit_l32
    },
}

SUPPORTED_MODELS = list(Models['Keras'].keys()) + list(Models['ViT-L'].keys())

Losses = {
    'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    'categorical_crossentropy': 'categorical_crossentropy'
}


class SwishActivation(Activation):

    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'


def swish(x, beta=1):
    return x * sigmoid(beta * x)


class NeuralNetFactory:

    def __init__(self, config: Config, input_shape: tuple, mode='Train', weights=None):
        """
        Artificial Neural network model factory

        Args:
            config (object): 
            input_shape:
        """
        assert config
        self.model_name = config.model
        self._logger = configure_logger(logfile_dir=config.model_dir, module=self.__class__.__name__)
        get_custom_objects().update({'swish': swish})

        if self.model_name not in SUPPORTED_MODELS:
            self._logger.debug(SUPPORTED_MODELS)
            raise RuntimeError(
                f"Supplied network name '{self.model_name}' is invalid. Please select one from the list.")

        assert os.path.exists(config.model_dir)

        self.config = config
        self.config_filepath = config.filepath
        self._model = None
        self.input_shape = input_shape
        self.pooling = config.pooling
        self.classes = config.num_classes
        self.activation = config.activation
        self.train_history = None
        self.weights = weights
        self.mode = mode
        self.callbacks = []
        self.model_dir = None
        self.y_true = None
        self.y_pred = None

        self._build(mode=self.mode)

    def _build(self, mode='Train'):
        self._logger.info(f"Building model, mode: {mode}")
        if mode == 'Train':
            if self.model_name in Models['Keras'].keys():
                self._model = Models['Keras'][self.model_name](
                    include_top=False,
                    input_shape=self.input_shape,
                    input_tensor=None,
                    pooling=self.pooling,
                    classes=self.classes,
                    classifier_activation=self.activation
                    )

            elif self.model_name in Models['ViT-L'].keys():
                self._model = Models['ViT-L'][self.model_name](
                    image_size=self.input_shape[0],
                    activation='sigmoid',
                    pretrained=False,
                    include_top=False,
                    pretrained_top=False,
                    classes=self.config.num_classes
                )

            self._append_classification_block()

        else:
            if not self.weights:
                raise RuntimeError(f"Unable to build model for '{mode}'. Must supply path to a valid .h5 model")

            self._model = keras.models.load_model(self.weights)

    def _append_classification_block(self):
        # Build the classification component of the network
        x = self._model.output
        x = BatchNormalization(name='BN1_Custom')(x)
        x = Dropout(0.7, name='Dropout1_Custom')(x)

        x = Dense(512, name='FC1')(x)
        x = BatchNormalization(name='BN2_Custom')(x)
        x = Activation(swish)(x)
        x = Dropout(0.5, name='Dropout_custom')(x)

        x = Dense(128, name='FC2')(x)
        x = BatchNormalization(name='BN3_Custom')(x)
        x = Activation(swish)(x)

        # output layer
        predictions = Dense(self.classes, activation="softmax")(x)
        self._model = Model(inputs=self._model.input, outputs=predictions)

    def _compile(self):
        # performance metrics
        metrics = ['acc']

        assert self.config.loss_function in Losses.keys()
        loss_function = Losses[self.config.loss_function]

        self._logger.debug(f"Compiling model, loss: {self.config.loss_function}, optimizer: {self.config.optimizer}")
        self._setup_callbacks()

        self._model.compile(
            loss=loss_function,
            optimizer=self.config.optimizer,
            metrics=metrics
        )

    def _setup_callbacks(self, benchmark=False):
        """
        Does 2 things.
            - Sets up the checkpoint_dir and log_dir
            - Sets up the callbacks
        """

        # Set up file system
        self._logger.debug(f"Setting up callbacks, model_dir: {self.config.model_dir}")

        model_dir = os.path.join(self.config.model_dir,
                                 str(date.today()),
                                 self.model_name,
                                 self.config.dataset.replace('/', '_'),
                                 self.config.optimizer)
        if benchmark:
            model_dir = os.path.join(model_dir, 'benchmark')
        if self.config.use_por:
            model_dir = os.path.join(model_dir, 'POR', f"{self.config.por_measure}")
        elif self.config.use_clo:
            model_dir = os.path.join(model_dir, 'CLO', f"{self.config.syllabus_measure}")
        else:
            model_dir = os.path.join(model_dir, "{}".format('baseline'))

        self._logger.debug(f'Model checkpoints and events: {model_dir}')

        while not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # print('Model Dir:', model_dir)
        self.model_dir = model_dir
        config_file_dump = os.path.join(model_dir,
                                        f"{self.model_name}_{self.config.dataset.replace('/', '_')}.ini")
        self.config.dump(config_file_dump)
        checkpoint_dir = os.path.join(model_dir, 'checkpoints')
        events_dir = os.path.join(model_dir, 'events')
        self._logger.debug('To open tensorboard run `tensorboard --logdir {}`'.format(events_dir))

        os.makedirs(events_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up callbacks for saving checkpoints, logging, and tensorboard
        best_model_path = os.path.join(checkpoint_dir, 'e{epoch:02d}.h5')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path,
                                                                 save_best_only=True,
                                                                 monitor='val_acc')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=events_dir,
                                                              histogram_freq=0,
                                                              update_freq='epoch')

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

        self.callbacks = [checkpoint_callback, tensorboard_callback, reduce_lr]

    def get(self, model_name):
        model_name = model_name.lower()

        if model_name == 'inception':
            self.input_shape = (75, 75, 3)

        if model_name in Models['Keras'].keys():
            self._model = Models['Keras'][model_name](
                include_top=False,
                input_shape=self.input_shape,
                input_tensor=None,
                pooling=self.pooling,
                classes=self.classes,
                weights=None)

        elif model_name in Models['ViT-L'].keys():
            self._model = Models['ViT-L'][model_name](
                image_size=self.input_shape[0],
                activation='sigmoid',
                pretrained=False,
                include_top=False,
                pretrained_top=False,
                classes=self.config.num_classes
            )
        else:
            raise RuntimeError('Unable to build model, supplied model name is does not exist')

        return self._model

    def train(self, dataset, benchmark: bool = False, epochs: int = None):
        """
        Train a model - with and without proposed preprocessing and optimization approaches

        Args:
            benchmark: benchmark training step - measure time
            epochs: number of epochs to run benchmark for
            dataset: ImageClassificationDataset - dataset to use for training

        Returns:

        """

        self._compile()

        if benchmark:
            assert epochs >= 1
            self._logger.info(f"Starting training benchmark, model: {self.model_name}, epochs: {epochs}...")
        else:
            epochs = self.config.epochs

        self._logger.debug(self.config.__repr__())

        self._logger.info(f"Starting training, model: {self.model_name}, "
                          f"use_por: {self.config.use_por}, "
                          f"POR Measure: {self.config.por_measure},"
                          f"use_syllabus: {self.config.use_clo}, "
                          f"Syllabus Measure: {self.config.syllabus_measure}...")

        # POR-Enabled training
        if self.config.use_por:

            assert self.config.por_measure != ''
            assert self.config.block_size >= 4
            assert self.config.rank_order in [0, 1]

            por = POR()
            por.measure = self.config.por_measure
            por.block_shape = (self.config.block_size, self.config.block_size)
            por.rank_order = self.config.rank_order

            train_dataset = dataset.train_dataset(batch_size=self.config.batch_size,
                                                  train_preprocessing=por.algorithm_por)
            validation_dataset = dataset.test_dataset(batch_size=self.config.batch_size)

            self.train_history = self._model.fit(
                train_dataset,
                epochs=epochs,
                batch_size=self.config.batch_size,
                validation_data=validation_dataset,
                verbose=1,
                callbacks=self.callbacks,
                shuffle=False
            )

        elif self.config.use_clo:
            assert self.config.syllabus_measure != ''

            curriculum = Curriculum()
            curriculum.measure = self.config.syllabus_measure
            curriculum.rank_order = self.config.rank_order
            curriculum.reference_image_index = None

            train_dataset = dataset.train_dataset(batch_size=self.config.batch_size,
                                                  train_preprocessing=curriculum.generate_syllabus,
                                                  clo=True)

            validation_dataset = dataset.test_dataset(batch_size=self.config.batch_size)

            self.train_history = self._model.fit(
                train_dataset,
                epochs=epochs,
                batch_size=self.config.batch_size,
                validation_data=validation_dataset,
                verbose=1,
                callbacks=self.callbacks,
                shuffle=False
            )

        else:
            train_dataset = dataset.train_dataset(batch_size=self.config.batch_size)
            validation_dataset = dataset.test_dataset(batch_size=self.config.batch_size)

            self.train_history = self._model.fit(
                train_dataset,
                epochs=epochs,
                batch_size=self.config.batch_size,
                validation_data=validation_dataset,
                verbose=1,
                callbacks=self.callbacks,
                shuffle=True
            )

        final_model_path = os.path.join(self.model_dir, 'FINAL.h5')
        self._logger.info(f"Final model saved to, {final_model_path}")
        self._model.save(final_model_path)

        self.evaluate(dataset.x_test, dataset.y_test, model_path=final_model_path, output_dir=self.model_dir)

    def _classification_report(self, output_dir=None):
        """

        Args:
            output_dir:

        Returns:

        """
        cr = classification_report(self.y_true, self.y_pred, output_dict=True)
        cr_df = pd.DataFrame.from_dict(cr)

        if output_dir:
            cr_df.to_csv(os.path.join(output_dir, f"{self.model_name}_{self.config.dataset}_cr.csv"))

    def evaluate(self, x_test, y_test, model_path=None, output_dir=None):
        """
        Evaluate generalization performance of final (best) model_path

        Args:
            x_test:
            y_test:
            model_path:
            output_dir:

        Returns:

        """
        sns.set_style('whitegrid')
        sns.set_palette("bright", 10, 1)

        if model_path:
            self._model = keras.models.load_model(model_path)

        self.y_pred = np.argmax(self._model.predict(x_test), axis=1)
        self.y_true = np.argmax(y_test, axis=1)
        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.ylabel("Groundtruth")
        plt.xlabel("Predicted")
        plt.title(f"{self.model_name.capitalize()} generalization confusion matrix")

        if output_dir:
            cm_file = os.path.join(output_dir, f"{self.model_name}_{self.config.dataset}_cm.png")
            plt.savefig(cm_file, dpi=1000)

        self._classification_report(output_dir=output_dir)
        plt.show()

    def timelined_benchmark(self, dataset, num_epochs=2, algorithm=None):

        # Initialize accumulators
        times_acc = dict()

        start_time = time.perf_counter()

        for epoch_num in tqdm.tqdm(range(1, num_epochs), desc=f"Benchmark"):
            epoch_enter = time.perf_counter()

            # Simulate training time
            train_enter = time.perf_counter()
            self.train(dataset=dataset, benchmark=True, epochs=epoch_num)
            time.sleep(0.1)
            train_elapsed = time.perf_counter() - train_enter

            # Record training information
            times_acc[epoch_num] = train_elapsed
            # epoch_elapsed = time.perf_counter() - epoch_enter

            # Record epoch information
            # times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)

        tf.print("Execution time:", time.perf_counter() - start_time)
        if not algorithm:
            return {"Baseline": times_acc}
        else:
            return {f"{algorithm}": times_acc}

    def plot(self):
        tf.keras.utils.plot_model(self._model)

    def open(self):
        model_path = f"{self.model_name}.h5"
        self._model.save(model_path)
        netron.start(model_path)

    def __repr__(self):
        return str(self._model.summary())
