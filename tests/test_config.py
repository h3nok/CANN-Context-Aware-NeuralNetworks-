from unittest import TestCase
from deepclo.config import Config


class TestConfig(TestCase):
    config = Config(config_file='deepclo_default_config.ini')
    print(config)

    def test_dump(self):
        self.config.dump(file='deepclo_default_config.ini')

    def test_get(self):
        print(self.config.dataset)
        print(self.config.epochs)
        print(self.config.learning_rate)
        print(self.config.loss_function)

