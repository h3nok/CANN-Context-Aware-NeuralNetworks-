from sqlalchemy import Column, String, Integer, Float, Boolean, text
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
import yaml

_base = declarative_base()


class TrainingFlags(_base):
    # Training hyperparameters model default inception_v2
    __tablename__ = 'training_params'
    __table_args__ = {"schema": "train_config"}

    id = Column(UUID(as_uuid=False), default=text("uuid_generate_v1()"), primary_key=True)
    name = Column(String, primary_key=True, nullable=False)
    dataset_name = Column(String, default='cifar10')
    purpose = Column(String, default="cifar")
    training_log_dir = Column(String, default=None)
    optimizer = Column(String, default='sgd')
    batch_size = Column(Integer, default=8)
    train_image_size = Column(Integer, default=224)
    max_number_of_steps = Column(Integer, default=10000000)
    log_every_n_steps = Column(Integer, default=10)
    save_interval_secs = Column(Integer, default=300)
    save_summaries_secs = Column(Integer, default=600)
    weight_decay = Column(Float, default=0.00004)
    adadelta_rho = Column(Float, default=0.95)
    adagrad_initial_accumulator_value = Column(Float, default=0.1)
    adam_beta1 = Column(Float, default=0.9)
    adam_beta2 = Column(Float, default=0.999)
    opt_epsilon = Column(Float, default=1.0)
    ftrl_learning_rate_power = Column(Float, default=-0.5)
    ftrl_initial_accumulator_value = Column(Float, default=0.1)
    ftrl_l1 = Column(Float, default=0.0)
    ftrl_l2 = Column(Float, default=0.0)
    momentum = Column(Float, default=0.9)
    rmsprop_momentum = Column(Float, default=0.9)
    rmsprop_decay = Column(Float, default=0.9)
    learning_rate_decay_type = Column(String, default='exponential')
    learning_rate = Column(Float, default=0.01)
    end_learning_rate = Column(Float, default=0.0001)
    label_smoothing = Column(Float, default=0.0)
    learning_rate_decay_factor = Column(Float, default=0.94)
    num_epochs_per_decay = Column(Float, default=2.0)
    sync_replicas = Column(Boolean, default=False)
    replicas_to_aggregate = Column(Integer, default=1)
    moving_average_decay = Column(Integer, default=None)
    tf_master = Column(String, default=None)
    num_clones = Column(Integer, default=1)
    clone_on_cpu = Column(Boolean, default=False)
    worker_replicas = Column(Integer, default=1)
    num_ps_tasks = Column(Integer, default=0)
    num_readers = Column(Integer, default=8)
    num_preprocessing_threads = Column(Integer, default=4)
    task = Column(Integer, default=1)
    labels_offset = Column(Integer, default=0)
    model_name = Column(String, default='inception_v2')
    preprocessing_name = Column(String, default=None)
    checkpoint_path = Column(String, default=None)
    checkpoint_exclude_scopes = Column(String, default=None)
    ignore_missing_vars = Column(Boolean, default=False)
    trainable_scopes = Column(String, default=None)
    dataset_split_name = Column(String, default=None)
    dataset_dir = Column(String, default=None)
    measure = Column(String, default=None)
    ordering = Column(Integer, default=0)
    patch_size = Column(Integer, default=8)
    curriculum = Column(Boolean, default=False)
    master = Column(String, default=None)
    backup_measure = Column(String, default=None)
    measure_list = Column(String, default=None)

    def to_dict(self):
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}

    @staticmethod
    def serialize(config):
        # dict to Hyperparams object
        assert config, "Supplied config is None"
        assert isinstance(config, dict), "Supplied config is not a dict instance\n"
        obj = TrainingFlags()
        for key, value in config.items():
            setattr(obj, key, value)

        return obj

    def dump(self, filename):
        try:
            with open(filename, 'w+') as outfile:
                yaml.dump(self.to_dict(), outfile, default_flow_style=False)
                outfile.close()
        except BaseException as e:
            print(e.args)


class EvalFlags:
    __tablename__ = 'eval_config'

    model = Column(String, default=None)
    checkpoint_path = Column(String, default=None)

    pass
