from config_parser import Config, ConfigType
from trainer_tf import Trainer, TrainerType

config_cifar10 = Config('/home/henok/research/configs/cifar10_baseline.yaml', ConfigType.Training)
config_cifar10_mi = Config('/home/henok/research/configs/cifar10_mi.yaml', ConfigType.Training)

config_cifar10_mi.load()
config_cifar10.load()
train_config = config_cifar10_mi.serialize()

trainer_type = TrainerType.tf
trainer = Trainer(train_config)
trainer.run(trainer_type)


