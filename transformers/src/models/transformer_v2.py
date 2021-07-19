import tensorflow as tf
from tensorflow import keras

class WellDecepticon(tf.keras.Model):

  def __init__(self, num_features, num_layers, initializer=None):
    super().__init__()
    if initializer is None:
      initializer = tf.keras.initializers.RandomNormal()
    layers = []
    for _ in range(num_layers):
      layers.append(WellDecepticonLayer(num_features=num_features, initializer=initializer))
    self.network = keras.Sequential(layers)

  def call(self, inputs):
    return self.network(inputs)


class WellDecepticonLayer(tf.keras.Model):

  def __init__(self, num_features, initializer=None):
    super().__init__()
    activation_input = tf.keras.layers.ReLU()
    #self.dropout = tf.keras.layers.Dropout(.2)
    #self.atl1 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    #self.atl2 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    #self.atl3 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    #self.atl4 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    #self.atl5 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    #self.atl6 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.sm   = keras.layers.Dense(units=num_features, activation='softmax', kernel_initializer=initializer)
    self.atf = keras.layers.Multiply()
    self.ff1 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff2 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff3 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff4 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff5 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff6 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.o   = keras.layers.Dense(units=1, activation=activation_input, kernel_initializer=initializer)

  def call(self, inputs):
    #tf.print(tf.shape(inputs))
    #x = self.dropout(inputs)
    #x = self.atl1(inputs)
    #x = self.atl2(x)
    #x = self.atl3(x)
    #x = self.atl4(x)
    #x = self.atl5(x)
    #x = self.atl6(x)
    x = self.sm(inputs)
    #tf.print(tf.shape(x))
    #inp= self.dropout(inputs)
    x = self.atf([x, inputs])
    x = self.ff1(x)
    x = self.ff2(x)
    x = self.ff3(x)
    x = self.ff4(x)
    x = self.ff5(x)
    x = self.ff6(x)
    x = self.o(x)
    #tf.print(tf.shape(x))
    return x

class AttentionTester(tf.keras.Model):

  def __init__(self, num_features, initializer=None):
    super().__init__()
    self.attention = tf.keras.layers.Attention()
    self.ff1 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff2 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff3 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff4 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff5 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.ff6 = keras.layers.Dense(units=num_features, activation=activation_input, kernel_initializer=initializer)
    self.o   = keras.layers.Dense(units=1, activation=activation_input, kernel_initializer=initializer)
  
  def call(self, inputs):
    pass
