from deepclo.config import Config
from deepclo.models.model_factory import NeuralNetFactory
from deepclo.pipe.dataset import ImageDataProvider
from deepclo.core.measures.measure_functions import RANK_MEASURES


def train_all(use_clo: bool = False,
              use_por: bool = False, rank_measures=None):
    """
    Bulk train models.

    Args:
        use_clo:
        use_por:
        rank_measures:

    Returns:

    """


    config = Config(config_file=args.config_file)
    dataset = ImageDataProvider(dataset_name=config.dataset,
                                custom_dataset_path=config.custom_dataset_path)

    if not use_clo and not use_por:
        config.use_clo = False
        config.use_por = False
        net = NeuralNetFactory(config=config,
                               input_shape=dataset.input_shape)
        net.train(dataset)
    else:
        config = Config(config_file=args.config_file)
        if not rank_measures:
            rank_measures = list(RANK_MEASURES.keys())

        print(f"Calling train_all() function, rank measures: {rank_measures} ... ")

        for m in rank_measures:
            config.use_clo = use_clo
            config.use_por = use_por
            if use_clo:
                config.syllabus_measure = m
            elif use_por:
                config.por_measure = m

            net = NeuralNetFactory(config=config, input_shape=dataset.input_shape)

            net.train(dataset)


def train(config_file):
    """
    Train based on a config file.
    """

    print("Calling train() function ... ")

    config = Config(config_file=config_file)
    dataset = ImageDataProvider(dataset_name=config.dataset,
                                custom_dataset_path=config.custom_dataset_path)

    net = NeuralNetFactory(config=config,
                           input_shape=dataset.input_shape)
    print(net)
    net.train(dataset)


def benchmark(config):
    dataset = ImageDataProvider(dataset_name=config.dataset,
                                custom_dataset_path=config.custom_dataset_path)
    net = NeuralNetFactory(config=config, input_shape=dataset.input_shape)

    return net.timelined_benchmark(dataset, num_epochs=config.epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Deep CLO experiment runs interface.')
    parser.add_argument('--benchmark', type=bool, default=False, help="Run training benchmark")
    parser.add_argument('--run_all', type=bool, default=True, help="Run bulk train operation")
    parser.add_argument('--rank_measures', type=list,
                        default=['KL', 'PSNR', 'MAX', 'CE', 'Entropy', 'MI'],
                        help="Run training benchmark")
    parser.add_argument('--config_file',
                        type=str,
                        default='../configurations/deepclo_train_config.ini',
                        required=True,
                        help="Path to .ini config file ")

    args = parser.parse_args()

    if args.benchmark:
        configuration = Config(config_file=args.config_file)
        results = benchmark(config=configuration)
    elif args.run_all:
        train_all(use_clo=True, use_por=False, rank_measures=args.rank_measures)
    else:
        train(args.config_file)
