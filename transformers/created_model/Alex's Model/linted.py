import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Layer

# import statement periodicly breaks, needs notebook to be restarted and sometimes for packages to be reinstalled
# from keras import backend as K
# variables
# todo: num fetures should be auto detected later based on array size, also these should be moved to a seperate file/ inputs
num_fetures = 13
num_epochs = 100
epoch_steps = 500
val_steps = 10
test_train_split = 0.2
random_seed = 10
# taking in variables to predict the median house price, sequence to single output analysis
# todo: bring in well data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=test_train_split, seed=random_seed
)
# turn data into a keras database
train_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_train, y_train, num_fetures,
)
val_dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
    x_test, y_test, num_fetures,
)
# index feture doesn't work, likely due to the way the sequential model is implemented/ optimized
# todo: fix getting the origional input for the dot product step
class FeedForwardAttention(Layer):
    def __init__(self):
        super(FeedForwardAttention, self).__init__()
        self.index = 0

    def build(self, input_shape):
        return

    def call(self, inputs):
        # print(self.index)
        # self.index = self.index  + 1
        return tf.matmul(
            inputs,
            tf.cast(tf.reshape(x_train[self.index], [num_fetures, 1]), tf.float32),
        )


# construction
model = keras.Sequential(
    [
        keras.layers.Dense(units=num_fetures, activation="relu"),
        keras.layers.Dense(units=num_fetures, activation="relu"),
        keras.layers.Dense(units=num_fetures, activation="relu"),
        # keras.layers.Dense(units=num_fetures, activation='softmax'),
        # not sure how this step might effect the back propagation
        # FeedForwardAttention(),
        keras.layers.Dense(units=num_fetures, activation="relu"),
        keras.layers.Dense(units=num_fetures, activation="relu"),
        keras.layers.Dense(units=num_fetures, activation="relu"),
        keras.layers.Dense(units=1, activation="relu"),
    ]
)
# run the model with a gradient decent optimizer and mean squared error
# Model seems to instantly fall into the same local minima, I was previously able to get it to train lower but not
# it doesn't seem to want to. I'll have to work further on this or just bring in the well data and hope for the best
# note: it falls into same local minima without the new model, I might be doing this wrong
model.compile(
    optimizer="SGD", loss=tf.keras.losses.MeanSquaredError(),
)
history = model.fit(
    train_dataset.repeat(),
    epochs=num_epochs,
    steps_per_epoch=epoch_steps,
    validation_data=val_dataset.repeat(),
    validation_steps=val_steps,
)
