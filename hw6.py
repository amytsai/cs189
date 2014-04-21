import scipy
import scipy.io as sio
import numpy as np
import code
import sys, traceback, pdb
import matplotlib.pyplot as plt
from math import *

def train_single_layer(images, labels, t_images, t_labels, isMeanSquare, epochs):
  """
  train a single layer neural network
  (mean square error)
  """
  # for graphing
  train_errors = []
  test_errors = []
  eps = []

  alpha = .008
  images_t = images.transpose()

  #convert labels to 10 dimensional vector
  t = np.zeros([60000, 10])
  for i in range(0, 60000):
    t_k = np.zeros(10)
    t_k[labels[i]] = 1
    t[i] = t_k
  t = t.transpose()

  #initialize weights to be +- ~10^-5
  W = np.random.rand(784, 10) * .00002
  bias = np.random.rand(1,10) * .002
  bias = bias - .001
  W = W - .00001

  for e in range(0, epochs):
    # take a mini batch
    sample = np.arange(60000)
    np.random.shuffle(sample)
    sample = sample[0:200]
    t_batch = t[:, sample]
    images_batch = images[:, sample]
    
    #forward pass
    W_t = W.transpose()
    S = np.add(np.dot(W_t, images_batch),bias.transpose())
    g_S = 1 / (1 + np.exp(-S))

    #calculate error every 10 epochs
    if e % 10 == 0:
      #training error
      temp = np.add(np.dot(W_t, images), bias.transpose())
      temp = 1/ (1 + np.exp(-temp))
      temp = temp > .5
      if isMeanSquare:
        temp = np.power(temp - t,  2)
        error = .1 * temp.sum()

      error = error/ 60000
      train_errors.append(error)
      
      #test error
      t_error = np.add(np.dot(W.transpose(), t_images), bias.transpose())
      t_g_S = 1 / (1 + np.exp(-t_error))
      t_output = np.argmax(t_g_S, 0)
      error_rate = np.not_equal(t_output, t_labels)
      error_rate = error_rate.sum()
      error_rate = error_rate / 10000.0
      test_errors.append(error_rate)
      eps.append(e)

      print 'error at epoch ', e, 'is: ', error

    #backwards pass
    if isMeanSquare:
      delta = (g_S - t_batch) * (1 - g_S)
    else:
      delta = (g_S - t_batch)
    a = alpha / pow(e + 1, .3)
    #update
    W = W - a * np.dot(images_batch, delta.transpose())
    bias = bias - a * delta.sum(1)

  plt.plot(eps, train_errors, 'g', eps, test_errors, 'b')
  plt.show()
  return W, bias

def predict_single_layer(images, labels, W, bias):
  #forward pass
  S = np.add(np.dot(W.transpose(), images), bias.transpose())
  g_S = 1 / (1 + np.exp(-S))
  output = np.argmax(g_S, 0)

  error_rate = np.not_equal(output, labels)
  error_rate = error_rate.sum()
  error_rate = error_rate / 10000.0
  print "TEST ERROR: ", error_rate
  return g_S, output

def main():
  """
  load data
  """
  #preprocessing
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

  test = sio.loadmat('test.mat')
  test = test['test']
  test_images = test['images']
  test_labels = test['labels']
  test_images = test_images[0][0]
  test_images = test_images.reshape(28*28, 10000)
  for col in range(0, test_images.shape[1]):
    column = test_images[:, col]
    column = column - np.average(column)
    column = column / np.std(column)
    test_images[:, col] = column
  test_labels = test_labels[0][0]
  test_labels = test_labels.flatten()

  W, bias = train_single_layer(images,labels, test_images, test_labels, True, 800)
  g_S, output = predict_single_layer(test_images, test_labels, W, bias)

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
