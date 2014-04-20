import scipy
import scipy.io as sio
import numpy as np
import code
from math import *

def train_single_layer(images, labels):
  """
  train a single layer neural network
  """
  #convert labels to 10 dimensional vector
  t = np.zeros([60000, 10])
  for i in range(0, 60000):
    t_k = np.zeros(10)
    t_k[labels[i][0]] = 1
    t[i] = t_k
  #initialize weights
  W = np.random.rand(784, 10)
  W_t = W.transpose()
  #forward pass
  S = np.dot(W_t, images)
  print np.shape(S)
  output = 1 / (1 + np.exp(S))
  return output
  #badkwards pass


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
  images = images / images.max()
  labels = labels[0][0]

  output = train_single_layer(images,labels)
  code.interact(local = locals())



if __name__ == "__main__":
  main()

import sys

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
