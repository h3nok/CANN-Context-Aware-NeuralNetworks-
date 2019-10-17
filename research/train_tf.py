from trainer_tf import Trainer
from configurations import TrainingConfig
import database_interface as dbi


sqlsess = dbi.SQLSession(host='hgh-d-22')
config = TrainingConfig(configfile="E:\\viNet_RnD\\Training\\Config\\viNet_2.2_E3_v2.yaml",
                        sqlsession=sqlsess)
t_params = config.params

e3_trainer = Trainer(config)
e3_trainer.run()
