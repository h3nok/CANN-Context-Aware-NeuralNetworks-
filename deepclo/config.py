from configparser import ConfigParser
import os


class Config:

    def __init__(self, config_file: str):
        """
        Training configuration

        Args:
            config_file: str - path to .ini config_file
        """

        assert config_file
        if not os.path.exists(config_file):
            print(f"Invalid config file path, path: '{config_file}'")
            self.dump()
        else:
            self._parser = ConfigParser()
            self.filepath = config_file
            self.parse()

    @property
    def default(self):
        self._parser = ConfigParser()
        self._parser.add_section('Training')
        self._parser.set('Training', 'dataset', 'cifar10')
        self._parser.set('Training', 'model_dir', r".")
        self._parser.set('Training', 'num_classes', '2')
        self._parser.set('Training', 'batch_size', '8')
        self._parser.set('Training', 'learning_rate', '0.001')
        self._parser.set('Training', 'optimizer', 'adam')
        self._parser.set('Training', 'loss_function', 'SparseCategoricalCrossentropy')
        self._parser.set('Training', 'use_clo', 'False')
        self._parser.set('Training', 'syllabus_measure', '')
        self._parser.set('Training', 'epochs', '100')
        self._parser.set('Training', 'use_por', 'False')
        self._parser.set('Training', 'block_shape', '8')
        self._parser.set('Training', 'por_measure', '')
        self._parser.set('Training', 'rank_order', '0')
        self._parser.set('Training', 'model', 'b0')
        self._parser.set('Training', 'activation', 'softmax')
        self._parser.set('Training', 'pooling', 'max')

        return self._parser

    @property
    def model(self):
        return self._parser.get(section='Training', option='model')

    @property
    def activation(self):
        return self._parser.get(section='Training', option='activation')

    @property
    def pooling(self):
        return self._parser.get(section='Training', option='pooling')

    @property
    def dataset(self):
        return self._parser.get(section='Training', option='dataset')

    @property
    def model_dir(self):
        return self._parser.get(section='Training', option='model_dir')

    @property
    def num_classes(self):
        return self._parser.getint(section='Training', option='num_classes')

    @property
    def batch_size(self):
        return self._parser.getint(section='Training', option='batch_size')

    @property
    def learning_rate(self):
        return self._parser.getfloat(section='Training', option='learning_rate')

    @property
    def epochs(self):
        return self._parser.getint(section='Training', option='epochs')

    @property
    def use_por(self):
        return self._parser.getboolean(section='Training', option='use_por')

    @property
    def block_size(self):
        return self._parser.getint(section='Training', option='block_shape')

    @property
    def rank_order(self):
        return self._parser.getint(section='Training', option='rank_order')

    @property
    def por_measure(self):
        return self._parser.get(section='Training', option='por_measure')

    @property
    def use_clo(self):
        return self._parser.getboolean(section='Training', option='use_clo')

    @property
    def syllabus_measure(self):
        return self._parser.get(section='Training', option='syllabus_measure')

    @property
    def loss_function(self):
        return self._parser.get(section='Training', option='loss_function')

    @property
    def optimizer(self):
        return self._parser.get(section='Training', option='optimizer')

    def parse(self):
        assert self.filepath
        assert os.path.exists(self.filepath), "Config file not found, path: {}".format(self.filepath)

        self._parser.read(self.filepath)

        return True

    def dump(self, file='deepclo_train_config.ini'):
        assert file
        with open(file, 'w+') as f:
            self.default.write(f)

    def __repr__(self):
        _config_str = "Config\n"
        for key, value in dict(self._parser.items('Training')).items():
            _config_str += f"\t{key}: {value}\n"

        return _config_str
