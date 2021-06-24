import toml
from dataclasses import dataclass

class Config:

    @classmethod
    def load(cls, file_name):
        with open(file_name) as f:
            data = toml.loads(f.read())
            return cls(**data)

    def __init__(self, **opts):
        self.train = TrainConfig(**opts["train"])
        self.model = ModelConfig(**opts["model"])

    def __getattribute__(self, name: str):
        return getattr(self.train, name)

class TrainConfig:

    batch_size = 10 
    eval_batch_size = 10

    forecast_window = 20 # How many values the model wil predict
    input_length = 20 # How many values the number will look at before making a predicition

    shift = 1
    epochs = 20

    device = "cpu"
    optimizer = "SGD"
    loss = "CrossEntropy"
    lr = 5.0 # Learning reate

    def __init__(self, **opts):
        for name, option in opts.items():
            setattr(self, name, option)

class ModelConfig:

    d_model = 1  # the dimension of the input vector (embedded vector)
    nhidden = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value

    def __init__(self, **opts):
        for name, option in opts.items():
            setattr(self, name, option)
