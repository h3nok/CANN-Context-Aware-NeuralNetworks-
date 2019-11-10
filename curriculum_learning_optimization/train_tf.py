from config_parser import Config, ConfigType
from trainer_tf import Trainer, TrainerType
from configurations import TrainingFlags

# config_cifar10 = Config('/home/henok/research/configs/cifar10_baseline.yaml', ConfigType.Training)
config_cifar10_mi = Config(r"E:\Thesis\configs\cifar10_mi.yaml", ConfigType.Training)
# training_config = TrainingFlags()
# training_config.dump(r"E:\Thesis\configs\cifar10_mi.yaml")
config_cifar10_mi.load()
# config_cifar10.load()
train_config = config_cifar10_mi.serialize()

trainer = Trainer(train_config)
trainer.run(TrainerType.slim)


