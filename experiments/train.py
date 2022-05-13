from deepclo.config import Config
from deepclo.models.model_factory import NeuralNetFactory
from deepclo.pipe.dataset import ImageDataProvider


def main(config):
    dataset = ImageDataProvider(dataset_name=config.dataset, custom_dataset_path=config.custom_dataset_path)
    net = NeuralNetFactory(config=config, input_shape=dataset.input_shape)
    net.train(dataset)


def benchmark(config):
    dataset = ImageDataProvider(dataset_name=config.dataset, custom_dataset_path=config.custom_dataset_path)
    net = NeuralNetFactory(config=config, input_shape=dataset.input_shape)
    return net.timelined_benchmark(dataset, num_epochs=config.epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Deep CLO experiment runs interface.')
    parser.add_argument('--benchmark', type=bool, default=False, help="Run training benchmark")
    parser.add_argument('--config_file',
                        type=str,
                        default='../configurations/deepclo_train_config.ini',
                        required=True,
                        help="Path to .ini config file ")

    args = parser.parse_args()
    configuration = Config(config_file=args.config_file)

    if args.benchmark:
        results = benchmark(config=configuration)
    else:
        main(config=configuration)

