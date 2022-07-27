from deepclo.config import Config
from deepclo.models.model_factory import NeuralNetFactory
from deepclo.pipe.dataset import ImageDataProvider


def main(args):
    """
    Train based on a config file.
    """

    config = Config(config_file=args.config_file)
    dataset = ImageDataProvider(dataset_name=config.dataset,
                                custom_dataset_path=config.custom_dataset_path)
    net = NeuralNetFactory(config=config,
                           input_shape=dataset.input_shape, mode='Test', weights=args.model_path)

    net.evaluate(x_test=dataset.x_test,
                 y_test=dataset.y_test,
                 output_dir=r'C:\deepclo\report\CIFAR10\Adam\b0')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Deep CLO test interface.')
    parser.add_argument('--config_file',
                        type=str,
                        default='../configurations/deepclo_eval_config.ini',
                        required=True,
                        help="Path to .ini config file ")
    parser.add_argument('--model_path', type=str,
                        default=r"C:\deepclo\training\2022-07-22\b0\cifar10\adam\baseline\checkpoints\e02.h5",
                        help='Path to a checkpoint file'
                        )

    args = parser.parse_args()
    main(args)
