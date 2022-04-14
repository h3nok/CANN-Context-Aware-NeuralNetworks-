from deepclo.config import Config
from deepclo.models.model_factory import NeuralNet
from deepclo.pipe.dataset import DeepCLODataProvider
from deepclo.utils import draw_timeline


def main(config):
    dataset = DeepCLODataProvider(dataset_name=config.dataset)
    net = NeuralNet(config=config, input_shape=dataset.input_shape)
    net.train(dataset)


def benchmark(config):
    dataset = DeepCLODataProvider(dataset_name=config.dataset)
    net = NeuralNet(config=config, input_shape=dataset.input_shape)

    return net.timelined_benchmark(dataset, num_epochs=config.epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--benchmark', type=bool, default=False, help="Run training benchmark")
    parser.add_argument('--config_file',
                        type=str,
                        default='../configurations/deepclo_train_config.ini',
                        help="Path to .ini config file ")

    args = parser.parse_args()
    configuration = Config(config_file=args.config_file)

    if args.benchmark:
        results = benchmark(config=configuration)
    else:
        main(config=configuration)

