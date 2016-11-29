"""
Simple feed-forward network for MNIST digit recognition
From https://github.com/Vict0rSch/deep_learning/blob/master/keras/feedforward/feedforward_keras_mnist.py
Also https://github.com/wxs/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb

To run:
import ex_mnist
data = ex_mnist.load_data() # Do this explicitly so we can use other data
model = ex_mnist.init_model()
(model, loss) = ex_mnist.run_network(data, model)
ex_mnist.plot_losses('loss.png', loss)
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import *
from keras.optimizers import RMSprop
from keras.datasets import mnist
from adv_util import AdversarialDataGenerator


class LossHistory(cb.Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    batch_loss = logs.get('loss')
    self.losses.append(batch_loss)


def load_data():
  print 'Loading data...'
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')

  X_train /= 255
  X_test /= 255

  y_train = np_utils.to_categorical(y_train, 10)
  y_test = np_utils.to_categorical(y_test, 10)

  X_train = np.reshape(X_train, (60000, 784))
  X_test = np.reshape(X_test, (10000, 784))

  print 'Data loaded.'
  return [X_train, X_test, y_train, y_test]


def init_model():
  start_time = time.time()
  print 'Compiling Model ... '
  layers = [
    Dense(500),
    Activation('relu'),
    Dropout(0.4),
    Dense(300),
    Activation('relu'),
    Dropout(0.4),
    Dense(10),
    Activation('softmax')
  ]

  ilayer = Input(shape=[784])
  olayer = reduce(lambda a,l: l(a), layers, ilayer)
  model = Model(ilayer, olayer)
  rms = RMSprop()
  model.compile(loss='categorical_crossentropy', optimizer=rms,
   metrics=['accuracy'])
  print 'Model compiled in {0} seconds'.format(time.time() - start_time)
  return model


def run_network(data, model, epochs=20, batch=256):
  """
  Use alternative to ImageDataGenerator
  """
  try:
    start_time = time.time()
    X_train, X_test, y_train, y_test = data

    history = LossHistory()

    print 'Attach adversarial example generator'
    datagen = AdversarialDataGenerator(model, epsilon=.01)
    print 'Training model...'
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch),
         samples_per_epoch=len(X_train), nb_epoch=epochs,
         callbacks=[history],
         validation_data=(X_test, y_test), verbose=2)

    print "Training duration : {0}".format(time.time() - start_time)
    score = model.evaluate(X_test, y_test, batch_size=16)

    print "Network's test score [loss, accuracy]: {0}".format(score)
    return model, history.losses
  except KeyboardInterrupt:
    print ' KeyboardInterrupt'
    return model, history.losses


def predict(model, images):
  """
  Takes an array of images. Obviously dimensions must match training set.
  """
  return model.predict_classes(images)


def display_classes(png, images, classes, ncol=4):
  """
  Draw a number of images and their predictions

  Example:
  images = data[1][:12]
  classes = model.predict_classes('classes.png', images)
  """
  fig = plt.figure()
  nrow = len(images) / ncol
  if len(images) % ncol > 0: nrow = nrow + 1

  def draw(i):
    plt.subplot(nrow,ncol,i)
    plt.imshow(images[i].reshape(28,28), cmap='gray', interpolation='none')
    plt.title('Predicted: %s' % classes[i])
  [ draw(i) for i in range(0,len(images)) ]
  plt.tight_layout()
  plt.savefig(png)

def plot_losses(png, losses):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(losses)
  ax.set_title('Loss per batch')
  plt.savefig(png)


if __name__ == '__main__':
  data = load_data() # Do this explicitly so we can use other data
  model = init_model()
  (model, loss) = run_network(data, model)
  plot_losses('ff_mnist_loss.png', loss)
