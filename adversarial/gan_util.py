import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from keras.datasets import mnist
#import seaborn as sns
from tqdm import tqdm

def load_mnist():
  img_rows, img_cols = 28, 28
  # the data, shuffled and split between train and test sets
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
  X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  X_train /= 255
  X_test /= 255
  return (X_train,y_train), (X_test,y_test)

# Freeze weights in the discriminator for stacked training
def set_trainable(net, val):
  net.trainable = val
  for l in net.layers:
    l.trainable = val

    
def train_gan(data, generator,discriminator,gan, nb_epoch=5000, plt_frq=25,BATCH_SIZE=32, loss=None):
  # Set up loss storage vector
  if loss is None: loss = {"d":[], "g":[]}

  #for e in tqdm(range(nb_epoch)):  
  for e in range(nb_epoch):
    # Make generative images
    image_batch = data[np.random.randint(0,data.shape[0],size=BATCH_SIZE),:,:,:]    
    noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
    generated_images = generator.predict(noise_gen)
    
    # Train discriminator on generated images
    X = np.concatenate((image_batch, generated_images))
    y = np.zeros([2*BATCH_SIZE,2])
    y[0:BATCH_SIZE,1] = 1
    y[BATCH_SIZE:,0] = 1
    
    #set_trainable(discriminator,True)
    d_loss  = discriminator.train_on_batch(X,y)
    loss["d"].append(d_loss)

    # train Generator-Discriminator stack on input noise to non-generated output class
    noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
    y2 = np.zeros([BATCH_SIZE,2])
    y2[:,1] = 1
    
    #set_trainable(discriminator,False)
    g_loss = gan.train_on_batch(noise_tr, y2 )
    loss["g"].append(g_loss)
      
    # Updates plots
    if e%plt_frq==plt_frq-1:
        plot_loss(loss, e)
        plot_gan_images(generator, e)
    return loss

def plot_loss(losses, idx=0):
  #display.clear_output(wait=True)
  #display.display(plt.gcf())
  plt.figure(figsize=(10,8))
  plt.plot(losses["d"], label='discriminitive loss')
  plt.plot(losses["g"], label='generative loss')
  plt.legend()
  #plt.show()
  plt.savefig("gan_loss_%s.png"%idx)

def plot_gan_images(generator, n_ex=16,dim=(4,4), figsize=(10,10), idx=0):
  noise = np.random.uniform(0,1,size=[n_ex,100])
  generated_images = generator.predict(noise)

  plt.figure(figsize=figsize)
  for i in range(generated_images.shape[0]):
    plt.subplot(dim[0],dim[1],i+1)
    img = generated_images[i,0,:,:]
    plt.imshow(img)
    plt.axis('off')
  plt.tight_layout()
  #plt.show()
  plt.savefig("gan_image_%s.png"%idx)


def plot_source_images(images, n_ex=16,dim=(4,4), figsize=(10,10), idx=0):
  idx = np.random.randint(0,images.shape[0],n_ex)
  generated_images = images[idx,:,:,:]

  plt.figure(figsize=figsize)
  for i in range(generated_images.shape[0]):
      plt.subplot(dim[0],dim[1],i+1)
      img = generated_images[i,0,:,:]
      plt.imshow(img)
      plt.axis('off')
  plt.tight_layout()
  #plt.show()
  plt.savefig("source_image_%s.png"%idx)

