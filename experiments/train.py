from deepclo.config import Config
from deepclo.models.model_factory import NeuralNetFactory
from deepclo.pipe.dataset import ImageDataProvider
from deepclo.core.measures.measure_functions import RANK_MEASURES

def main(use_clo: bool = False, use_por: bool = False):
    config = Config(config_file=args.config_file)
    dataset = ImageDataProvider(dataset_name=config.dataset, custom_dataset_path=config.custom_dataset_path)

    if not use_clo and not use_por:
        config.use_clo = False
        config.use_por = False
        net = NeuralNetFactory(config=config, input_shape=dataset.input_shape)
        net.train(dataset)
    else:
        config = Config(config_file=args.config_file)

        for m in RANK_MEASURES.keys():
            config.use_clo = use_clo
            config.use_por = use_por
            config.epochs = 1

            if use_clo:
                config.syllabus_measure = m
            elif use_por:
                config.por_measure = m

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

    if args.benchmark:
        configuration = Config(config_file=args.config_file)
        results = benchmark(config=configuration)
    else:
        main(use_clo=True, use_por=False)

