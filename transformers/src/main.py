"""
Runs through training and plotting of the loss function for the current running model
"""
from collections import Counter, OrderedDict
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import os
import toml

import tensorflow as tf
from tensorflow import optimizers as optim
from keras import metrics
from keras import losses
from keras import layers
import keras

import toml
import os

from dataloader import WellLogDataset
from utils import Config

def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)

    pass

def test_step(model, x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)

    test_loss(loss)


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

    x_train = [ d[0] for d in train_data]
    y_train = [ d[1] for d in train_data]

    x_test =  [ d[0] for d in test_data]
    y_test =  [ d[1] for d in test_data]

    # Define our TransformerModel
    model = keras.Sequential([
        layers.Dense(units=2, activation="relu"),
        layers.Dense(units=10, activation="relu"),
        layers.Dense(units=10, activation="relu"),
        layers.Dense(units=1, activation="relu"),
    ])

    # Loss function 

    loss_object = None
    if cnf.loss == "MSE":
        loss_object = losses.MeanSquaredError()
    if cnf.loss == "CrossEntropy":
        loss_object = losses.BinaryCrossentropy()

    lr = cnf.lr

    # Loss Metrics
    test_loss = metrics.Mean()
    train_loss = metrics.Mean()

    optimizer = None
    if cnf.optimizer == "SGD":
        optimizer = optim.SGD(learning_rate=lr)
    if cnf.optimizer == "Adam":
        optimizer = optim.Adam(learning_rate=lr)

    assert optimizer
    assert train_loss
    assert test_loss

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_log_dir = f"{cnf.exp_dir}/logs/"
    test_log_dir  = f"{cnf.exp_dir}/logs/"

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer  = tf.summary.create_file_writer(test_log_dir)

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 1.0, gamma=0.95  # type:ignore
    # )

    # # Training loop
    # for epoch in range(1, cnf.epochs + 1):
    #     epoch_start_time = time.time()
    #     train(model, optimizer, criterion, ntokens, train_data, cnf)
    #     val_loss = evaluate(model, criterion, ntokens, val_data, cnf)
    #     print("-" * 89)
    #     print(
    #         "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
    #         "valid ppl {:8.2f}".format(
    #             epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
    #         )
    #     )
    #     print("-" * 89)
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         best_model = model

    for epoch in range(cnf.epochs):
        for (x_train, y_train) in train_dataset:
            train_step(model, optimizer, x_train, y_train)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            # tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for (x_test, y_test) in test_dataset:
            test_step(model, x_test, y_test)
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            # tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
      
        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print (template.format(epoch+1,
                             train_loss.result(), 
                             # train_accuracy.result()*100,
                             test_loss.result(), 
                             # test_accuracy.result()*100,
                             ))

    # # Export results into dataframe (slices are used due to size differences)
    # results = pd.DataFrame(
    #     {
    #         "Output": [t.item() for t in full_output.view(-1)],
    #         "Input": conv(test_data.view(-1), vocab)[105:],
    #     },
    #     index=pd.Index(name="depth", data=well[1250:3899:5].index.values[107:]),
    # )

    # # Lets plot the results to see the final output
    # fig = plt.figure(figsize=(8, 10))
    # x = well["gamma"].values

    # ax = plt.gca()
    # ax.plot(results["Output"], results.index.values, label="Prediction", color="red")
    # ax.plot(
    #     well[:1000:5]["gamma"], well[:1000:5].index.values, label="Train", color="gray"
    # )
    # ax.plot(
    #     well[1000:1250:5]["gamma"],
    #     well[1000:1250:5].index.values,
    #     label="Validation",
    #     color="yellow",
    # )
    # ax.plot(
    #     well[1250::5]["gamma"], well[1250::5].index.values, label="Target", color="blue"
    # )
    # ax.legend()
    # ax.invert_yaxis()
    # plt.savefig(f"results_{cnf.data.year}.png")
    # plt.show()
