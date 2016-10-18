# https://github.com/Vict0rSch/deep_learning/tree/master/keras/feedforward
# http://blog.fastforwardlabs.com/post/139921712388/hello-world-in-keras-or-scikit-learn-versus

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

model = Sequential()
model.add(Dense(10, input_shape=(2,)))
model.add(Activation('tanh'))
model.add(Dense(1))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(train_X, train_y_ohe, verbose=0, batch_size=1)

loss, accuracy = model.evaluate(test_X, test_y_ohe, show_accuracy=True, verbose=0)
print("Test fraction correct (Accuracy) = {:.2f}".format(accuracy))
