from config_parser import Config, ConfigType
from trainer_tf import Trainer, TrainerType
from configurations import TrainingFlags

# config_cifar10 = Config(r'config/cifar10_baseline.yaml', ConfigType.Training)
config_cifar10_mi = Config(r"config/cifar10_mi.yaml", ConfigType.Training)

config_cifar10_mi.load()
# config_cifar10.load()

train_config_mi = config_cifar10_mi.serialize()
trainer = Trainer(train_config_mi)
trainer.run(TrainerType.slim)


