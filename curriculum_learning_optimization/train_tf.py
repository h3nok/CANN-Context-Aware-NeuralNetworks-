import database_interface as dbi
from configurations import TrainingConfig

from trainer_tf import Trainer

sqlsess = dbi.SQLSession(host='hgh-d-22')
config = TrainingConfig(configfile="E:\\viNet_RnD\\Training\\Config\\cifar10_train_config.yaml",
                        sqlsession=sqlsess)
t_params = config.params

e3_trainer = Trainer(config)
e3_trainer.run()
