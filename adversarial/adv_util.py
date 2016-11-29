import numpy as np
from numpy.random import uniform
from keras import backend as K
from keras.preprocessing.image import NumpyArrayIterator, DirectoryIterator



class AdversarialDataGenerator(object):
  '''Generate minibatches with real-time adversarial training.
  Requires a reference to the model in order to access the loss for the 
  training sample.

  '''
  def __init__(self,
         model,
         epsilon,
         prob=0.2,
         dim_ordering='default'):
    if dim_ordering == 'default':
      dim_ordering = K.image_dim_ordering()
    self.__dict__.update(locals())
    self.model = model
    self.epsilon = epsilon
    self.prob = prob
    self.idx = 0

    if dim_ordering not in {'tf', 'th'}:
      raise Exception('dim_ordering should be "tf" (channel after row and '
              'column) or "th" (channel before row and column). '
              'Received arg: ', dim_ordering)
    self.dim_ordering = dim_ordering
    if dim_ordering == 'th':
      self.channel_index = 1
      self.row_index = 2
      self.col_index = 3
    if dim_ordering == 'tf':
      self.channel_index = 3
      self.row_index = 1
      self.col_index = 2


  def flow(self, X, y, batch_size=32, shuffle=True, seed=None,
       save_to_dir=None, save_prefix='', save_format='jpeg'):
    """
    Basic flow (haha) is inside the NumpyArrayIterator, which calls
    random_transform followed by standardize.
    """
    return NumpyArrayIterator(
      X, y, self,
      batch_size=batch_size, shuffle=shuffle, seed=seed,
      dim_ordering=self.dim_ordering,
      save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

  def flow_from_directory(self, directory,
              target_size=(256, 256), color_mode='rgb',
              classes=None, class_mode='categorical',
              batch_size=32, shuffle=True, seed=None,
              save_to_dir=None, save_prefix='', save_format='jpeg'):
    return DirectoryIterator(
      directory, self,
      target_size=target_size, color_mode=color_mode,
      classes=classes, class_mode=class_mode,
      dim_ordering=self.dim_ordering,
      batch_size=batch_size, shuffle=shuffle, seed=seed,
      save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

  def standardize(self, x):
    """
    Called by the NumpyArrayIterator after random_transform. 
    For now just return x to keep things simple.
    """
    return x

  def random_transform(self, x):
    """
    Called by the NumpyArrayIterator. This is where the adversarial
    training implementation resides.

    https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

    @param x A single image, possibly with multiple channels
    """
    self.idx = self.idx + 1
    if uniform() > self.prob:
      if self.idx % 100 == 0:
        print "[%s] Skipped adversarial example"%self.idx
      return x

    x = np.expand_dims(x,axis=0)
    ilayer = self.model.layers[0]
    olayer = self.model.layers[-2]
    img = ilayer.input
    # The actual optimized objective is the mean of the output array across 
    # all datapoints
    # https://keras.io/objectives/
    loss = K.mean(olayer.output)
    #import pdb;pdb.set_trace()
    # http://deeplearning.net/software/theano/tutorial/examples.html
    grad = K.gradients(loss, img)[0]
    fn = K.function([img, K.learning_phase()], [loss, grad])
    gx = fn([x,0])

    ax = x + self.epsilon * np.sign(gx[1])

    if self.idx % 100 == 0:
      print "[%s] Got loss of %s" % (self.idx,gx[0])

    return ax

  def fit(self, X, augment=False, rounds=1, seed=None):
   """
   Unused
   """
   pass
