import yaml
from enum import Enum
import os
from configurations import TrainingFlags


class ConfigType(Enum):
    Training = "Training Configuration"
    Evaluation = "Evaluation Configuration"
    Testing = "Testing Configuration"


class Config:
    type = None
    config_dict = dict()

    def __init__(self, file, type : ConfigType):
        assert os.path.exists(file), "Supplied file doesn't exist!"

        self.config_file = file
        self.type = type

    def load(self):
        try:
            with open(self.config_file) as file:
                self.config_dict = yaml.load(file, Loader=yaml.FullLoader)
                file.close()
                print("Successfully loaded configuration file \n")
        except BaseException as e:
            print(e.args)

    @property
    def get(self):
        return self.config_dict

    def pprint(self):
        for key, value in self.config_dict.items():
            print("{}: {}".format(key, value))

    def serialize(self):
        if self.type is ConfigType.Training:
            return TrainingFlags.serialize(self.config_dict)
        else:
            raise RuntimeError("Unknown config type")

    def dump(self, filename):
        assert self.config_dict
        try:
            with open(filename, 'w+') as outfile:
                yaml.dump(self.config_dict, outfile, default_flow_style=False)
                outfile.close()
        except BaseException as e:
            print(e.args)

