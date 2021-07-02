"""
Contains the Config class to setup the system
"""
import json
import toml


class Config:
    """
    The full configuation system that defines
    training
    dataset
    and model hyperparmeters
    """
    @classmethod
    def load_toml(cls, file_name):
        """
        Import the configuration using a toml file at the specified location
        Example configuration:
        ```toml
            [model]
            d_model = 200  # the dimension of the input vector (embedded vector)
            nhidden = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
            nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            nhead = 4  # the number of heads in the multiheadattention models
            dropout = 0.2  # the dropout value

            [data]
            year = 2018 # The specificed year dataset to train on

            [train]
            batch_size = 10
            eval_batch_size = 10

            forecast_window = 1 # How many values the model wil predict
            input_length = 10 # How many values the number will look at before making a predicition

            epochs = 20

            device = "cpu"
            optimizer = "SGD"
            loss = "CrossEntropy"
            lr = 5.0 # Learning reate
        ```
        """
        with open(file_name) as f:
            data = toml.loads(f.read())
            return cls(**data)

    @classmethod
    def load_json(cls, file_name):
        """
        Import the configuration using a toml file at the specified location
        Example configuration:
        ```json
        {
          "data": {
            "year": 2018
          },
          "model": {
            "d_model": 200,
            "dropout": 0.2,
            "nhead": 4,
            "nhidden": 200,
            "nlayers": 2
          },
          "train": {
            "batch_size": 10,
            "device": "cpu",
            "epochs": 20,
            "eval_batch_size": 10,
            "forecast_window": 1,
            "input_length": 10,
            "loss": "CrossEntropy",
            "lr": 5.0,
            "optimizer": "SGD"
          }
        }
        ```
        """
        with open(file_name) as f:
            data = json.loads(f.read())
            return cls(**data)

    def __init__(self, **opts):
        self.train = TrainConfig(**opts["train"])
        self.model = ModelConfig(**opts["model"])
        self.data = DataConfig(**opts["data"])

    def __getattr__(self, name: str):
        """
        Training config options will be used the most
        Lets make it not so verbose to do
        `cnf.train.{option}` and do `cnf.option` instead
        """
        return getattr(self.train, name)


class GeneralConfig:
    """
    Base Class for subsettings, this is used so we won't
    HAVE to update the python file when new options are added
    only the config file

    It is still best practice to put the name of each along with
    a default value
    """
    def __init__(self, **opts):
        for name, option in opts.items():
            setattr(self, name, option)


class DataConfig(GeneralConfig):
    """
    Configuration for the Dataset
    TODO: Soon add Test/train split or more
    specific parameters for parsing
    """

    year: int


class TrainConfig(GeneralConfig):
    """
    The configuration options for Training
    """

    batch_size = 10
    eval_batch_size = 10

    forecast_window = 20  # How many values the model wil predict
    input_length = (
        20  # How many values the number will look at before making a predicition
    )

    shift = 1
    epochs = 20

    device = "cpu"
    optimizer = "SGD"
    loss = "CrossEntropy"
    lr = 5.0  # Learning reate


class ModelConfig(GeneralConfig):
    """
    The configuration options for the model
    """

    d_model = 1  # the dimension of the input vector (embedded vector)
    nhidden = (
        200  # the dimension of the feedforward network model in nn.TransformerEncoder
    )
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
