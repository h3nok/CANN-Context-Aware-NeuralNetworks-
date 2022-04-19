import os
import shutil
import time
import netron
import tqdm

import tensorflow as tf

from vit_keras import vit
from deepclo.algorithms.curriculum import Curriculum
from deepclo.config import Config

from deepclo.utils import configure_logger
from deepclo.algorithms.por import POR
from classification_models.tfkeras import Classifiers

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
        'effl2': keras_apps.EfficientNetV2L,
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
    'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
}


class NeuralNet:

    def __init__(self, config: Config, input_shape: tuple):
        """
        Artificial Neural network model factory

        Args:
            config (object): 
            input_shape:
        """
        assert config
        self.model_name = config.model
        self._logger = configure_logger(logfile_dir=config.model_dir, module=self.__class__.__name__)

        if self.model_name not in SUPPORTED_MODELS:
            self._logger.debug(SUPPORTED_MODELS)
            raise RuntimeError(
                f"Supplied network name '{self.model_name}' is invalid. Please select one from the list.")

        self.config = config
        self.config_filepath = config.filepath
        self._model = None
        self.input_shape = input_shape
        self.pooling = config.pooling
        self.classes = config.num_classes
        self.activation = config.activation
        self.train_history = None
        self.callbacks = []

        self._build()

    def _build(self):
        if self.model_name in Models['Keras'].keys():
            self._model = Models['Keras'][self.model_name](include_top=False,
                                                           input_shape=self.input_shape,
                                                           input_tensor=None,
                                                           pooling=self.pooling,
                                                           classes=self.classes,
                                                           classifier_activation=self.activation)
        elif self.model_name in Models['ViT-L'].keys():
            self._model = Models['ViT-L'][self.model_name](
                image_size=self.input_shape[0],
                activation='sigmoid',
                pretrained=False,
                include_top=False,
                pretrained_top=False,
                classes=self.config.num_classes
            )

    def _setup_callbacks(self, benchmark=False):
        """
        Does 2 things.
            - Sets up the checkpoint_dir and log_dir
            - Sets up the callbacks
        """

        # Set up file system
        self._logger.debug(f"Setting up callbacks, model_dir: {self.config.model_dir}")

        model_dir = os.path.join(self.config.model_dir, self.model_name, self.config.dataset, self.config.optimizer)
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

        config_file_dump = os.path.join(model_dir, f"{self.model_name}_{self.config.dataset}.ini")
        shutil.copy(self.config_filepath, config_file_dump)
        checkpoint_dir = os.path.join(model_dir, 'checkpoints')
        events_dir = os.path.join(model_dir, 'events')
        self._logger.debug('To open tensorboard run `tensorboard --logdir {}`'.format(events_dir))

        os.makedirs(events_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set up callbacks for saving checkpoints, logging, and tensorboard
        best_model_path = os.path.join(checkpoint_dir, 'e{epoch:02d}.h5')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path,
                                                                 save_weights_only=True,
                                                                 verbose=1,
                                                                 save_freq='epoch',
                                                                 save_best_only=True,
                                                                 mode='min'
                                                                 )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=events_dir,
                                                              histogram_freq=0,
                                                              update_freq='epoch')

        self.callbacks = [checkpoint_callback, tensorboard_callback]

    def train(self, dataset, benchmark: bool = False, epochs: int = None):
        """
        Train a model - with and without proposed preprocessing and optimization approaches

        Args:
            benchmark: benchmark training step - measure time
            epochs: number of epochs to run benchmark for
            dataset: ImageClassificationDataset - dataset to use for training

        Returns:

        """
        # performance metrics
        metrics = ['acc', 'mse']

        assert self.config.loss_function in Losses.keys()
        loss_function = Losses[self.config.loss_function]

        self._logger.debug(f"Compiling model, loss: {self.config.loss_function}, optimizer: {self.config.optimizer}")
        self._setup_callbacks()

        self._model.compile(
            loss=loss_function,
            optimizer=self.config.optimizer,
            metrics=metrics
        )

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
