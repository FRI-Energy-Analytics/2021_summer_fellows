"""
Runs through training and plotting of the loss function for the current running model
"""
from collections import Counter, OrderedDict
import math
from models.transformer_v2 import WellDecepticon
from models.transformer_v2 import WellDecepticonLayer
import time

import matplotlib.pyplot as plt
import pandas as pd
import os
import toml

import tensorflow as tf
from tensorflow import optimizers as optim
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import toml
import os

from dataloader import WellLogDataset
from utils import Config

if __name__ == "__main__":

    # TODO: make this config file be through command line
    config_file = "options/config.toml"
    cnf = Config.load_toml(config_file)

    if os.path.exists(cnf.exp_dir):
	    raise Exception('Oops... {} already exists'.format(cnf.exp_dir))
    os.makedirs(cnf.exp_dir)

    with open(os.path.join(cnf.exp_dir, 'config.toml'), 'w') as f:
        toml.dump(cnf.raw_data, f)

    log_dir = os.path.join(cnf.exp_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Data split
    # TODO: Make root_dir be passed in via command line option
    data = WellLogDataset("export_csv", cnf)
    train_size = int(len(data) * cnf.data.train_split)
    test_size = len(data) - train_size

    train_data = [data[i] for i in range(train_size)]
    test_data =  [data[i] for i in range(train_size, train_size + test_size -1)]
    #print(train_size, "train_size\n\n\n\n\n\n")
    #train_data = [data[i] for i in range(100)]
    #test_data =  [data[i] for i in range(100, 200)]
    #print("test\n\n\n\n\n\n")

    x_train = np.array([ d[0] for d in train_data])
    y_train = np.array([ d[1] for d in train_data])

    print("X", x_train.shape)
    print("Y", y_train.shape)


    assert not np.any(np.isnan(x_train))
    assert not np.any(np.isnan(y_train))

    x_test =  [ d[0] for d in test_data]
    y_test =  [ d[1] for d in test_data]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    #     path="boston_housing.npz", test_split=0.5, seed=10
    # )

    # train_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    #     x_train, y_train, cnf.input_length, #batch_size = 1
    # )

    # test_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    #     x_test, y_test, cnf.input_length, #batch_size = 1
    # )
        
    # Define our TransformerModel
    #model = WellDecepticon(cnf.input_length, num_layers=2)
    model = WellDecepticonLayer(cnf.input_length, output_size=cnf.forecast_window, initializer=tf.keras.initializers.RandomNormal())

    # Loss function 

    loss = cnf.loss
    if cnf.loss == "MSE":
        loss = tf.keras.losses.MeanSquaredError()
    if cnf.loss == "CrossEntropy":
        loss = tf.keras.losses.BinaryCrossentropy()

    lr = cnf.lr

    optimizer = None
    if cnf.optimizer == "SGD":
        optimizer = optim.SGD(learning_rate=lr)
    if cnf.optimizer == "Adam":
        optimizer = optim.Adam(learning_rate=lr)
    if cnf.optimizer == "Nadam":
        optimizer = optim.Nadam(learning_rate=lr)

    assert optimizer
    assert loss


    log_dir = f"{cnf.exp_dir}/logs/"

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 1.0, gamma=0.95  # type:ignore
    # )
    model.compile(
        optimizer=optimizer,
        loss=loss
    )

    history = model.fit(
        train_dataset.repeat(),
        epochs=cnf.epochs,
        steps_per_epoch=1000,
        validation_data=test_dataset.repeat(),
        validation_steps=1000,
        callbacks=[tensorboard_callback]
    )

    # Plot full well
    id = 0
    start = 0
    well = data.wells.loc[int(str(id) + str(cnf.data.year))]
    depth = well["Depth"].values
    gamma = well["Gamma"].values
    _input = gamma[start : start + cnf.input_length]
    _input = _input.reshape(-1)
    well_length = len(depth) - (len(depth) % (cnf.input_length + cnf.forecast_window))
    full_output = gamma[start:start + cnf.input_length]
    for i in tqdm(range(len(depth) - cnf.input_length)):
        eval_y = model(np.expand_dims(_input, -1))
        _input = np.append(_input[1::], tf.reduce_sum(eval_y, 1).numpy()[-cnf.forecast_window])
        full_output = np.append(full_output, tf.reduce_sum(eval_y).numpy())
        start += cnf.forecast_window
    plt.plot(full_output, depth, label="Output", color="red")
    plt.plot(gamma,       depth, label="Target",  color="blue")
    plt.show()

        
    # Plot Individual Wells
    ids = range(len(data))
    for id in ids:
        test_x, test_y = data[id]  # Grab example to validate
        depth = data.wells.loc[int(str(id) + str(cnf.data.year))]["Depth"].values

        eval_y = tf.reduce_sum(model(np.expand_dims(test_x, -1)), 1)

        depth_input = depth[:cnf.input_length]
        depth_output = depth[cnf.forecast_window:cnf.input_length + cnf.forecast_window]

        plt.plot(test_y, depth_output, label="Target", color="orange") # Target
        plt.plot(eval_y, depth_output, label="Output", color="red") # Output

        plt.plot(test_x, depth_input, label="Input", color="blue") # Input
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
