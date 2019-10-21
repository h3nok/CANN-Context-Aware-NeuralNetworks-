from config_parser import Config, ConfigType
from trainer_tf import Trainer

config_file = Config('/home/henok/research/configs/cifar10.yaml', ConfigType.Training)
config_file.load()

train_config = config_file.serialize()

trainer = Trainer(train_config)
trainer.run()


