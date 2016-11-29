"""
http://ml4a.github.io/guides/convolutional_neural_networks/

To run:
import cnn_mnist
data = cnn_mnist.load_data() # Do this explicitly so we can use other data
model = cnn_mnist.init_model()
(model, loss) = cnn_mnist.run_network(data, model)
cnn_mnist.plot_losses('loss.png', loss)
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.datasets import mnist

BATCH_SIZE = 256
# (channels, width, height)
INPUT_SHAPE = (1,28,28)

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

    X_train = np.reshape(X_train, (60000, 1, 28,28))
    X_test = np.reshape(X_test, (10000, 1, 28,28))

    print 'Data loaded'
    return [X_train, X_test, y_train, y_test]


def init_model():
    """
    """
    start_time = time.time()
    print 'Compiling model...'
    model = Sequential()

    model.add(Convolution2D(64, 3,3, border_mode='valid', input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.25))

    model.add(Flatten())

    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms,
      metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)

    model.summary()
    return model


def init_model_1():
    """
    Uses functional API to construct model.
    https://keras.io/models/model/
    """
    start_time = time.time()
    print 'Compiling model...'
    layers = [
      Convolution2D(64, 3,3, border_mode='valid')
      Activation('relu'),
      Convolution2D(64, 3,3, border_mode='valid'),
      Activation('relu'),
      MaxPooling2D(pool_size=(2,2)),
      Dropout(.25),
      Flatten(),
      Dense(128),
      Activation('relu'),
      Dropout(.5),
      Dense(10),
      Activation('softmax')
    ]

    ilayer = Input(shape=INPUT_SHAPE),
    olayer = reduce(lambda a,l: l(a), layers, ilayer)
    model = Model(ilayer, olayer)
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms,
      metrics=['accuracy'])
    print 'Model compiled in {0} seconds'.format(time.time() - start_time)
    return model


def run_network(data=None, model=None, epochs=20, batch=BATCH_SIZE):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print 'Training model...'
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
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
  plot_losses('cnn_mnist_loss.png', loss)
  
