from unittest import TestCase
from deepclo.models.model_factory import NeuralNetFactory
from deepclo.config import Config


class TestNeuralNet(TestCase):
    config = Config('../configurations/deepclo_train_config.ini')
    net_factory = NeuralNetFactory(config=config, input_shape=(32, 32, 3))

    def test_get(self):
        models = ['nasnet', 'densenet',
                  'mobilenet'
                  ]

        for name in models:
            model = self.net_factory.get(name)
            print(f"{name}: {model.count_params()}")