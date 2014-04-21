import scipy
import scipy.io as sio
import numpy as np
import code
import sys, traceback, pdb
from math import *

def train_single_layer(images, labels, epochs):
  """
  train a single layer neural network
  """
  alpha = .07
  images_t = images.transpose()
  #convert labels to 10 dimensional vector
  t = np.zeros([60000, 10])
  for i in range(0, 60000):
    t_k = np.zeros(10)
    t_k[labels[i]] = 1
    t[i] = t_k
  t = t.transpose()
  #initialize weights
  W = np.random.rand(784, 10) * .002
  W = W - .001

  for e in range(0, epochs):
    sample = np.arange(60000)
    np.random.shuffle(sample)
    sample = sample[0:200]
    t_batch = t[:, sample]
    images_batch = images[:, sample]
    W_t = W.transpose()
    #forward pass
    S = np.dot(W_t, images_batch)
    g_S = 1 / (1 + np.exp(-S))
    output = g_S > .5

    #calculate error every 10 epochs
    if e % 10 == 0:
      temp = np.power(output - t_batch,  2)
      error = .1 * temp.sum()
      print 'error at epoch ', e, 'is: ', error
      print 'min g_S', g_S.min()
      print 'max g_S', g_S.max()

    #backwards pass
    delta = (t_batch - g_S) * g_S * (1 - g_S)
    a = alpha / pow(e + 1, .5)
    W = W - a * np.dot(images_batch, delta.transpose())

  return W

def main():
  """
  load data
  """
  training = sio.loadmat('train.mat')
  training = training['train']
  images = training['images']
  labels = training['labels']

  images = images[0][0]
  images = images.reshape(28*28, 60000)
  for col in range(0, images.shape[1]):
    column = images[:, col]
    column = column - np.average(column)
    column = column / np.std(column)
    images[:, col] = column
  labels = labels[0][0]
  labels = labels.flatten()

  output = train_single_layer(images,labels,10)
  code.interact(local = locals())



if __name__ == "__main__":
  try: 
    main()

  except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

def info(type, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type != AssertionError:
      # we are in interactive mode or we don't have a tty-like
      # device, so we call the default hook
      sys.__excepthook__(type, value, tb)
  else:
      import traceback, pdb
      # we are NOT in interactive mode, print the exception...
      traceback.print_exception(type, value, tb)
      print
      # ...then start the debugger in post-mortem mode.
      pdb.pm()

sys.excepthook = info
