import scipy
import scipy.io as sio
import numpy as np
import code
import sys, traceback, pdb
import matplotlib.pyplot as plt
from math import *

def train_single_layer_ms(images, labels, t_images, t_labels, epochs):
  """
  train a single layer neural network
  using mean square error
  """
  #preprocessing specific to mean square
  means = np.average(images, 1).reshape(784, 1)
  images = np.subtract(images, means)
  stds = np.std(images, 1)
  for col in range(0, images.shape[0]):
    column = images[col, :]
    if stds[col] != 0.0:
      column = column / stds[col]
    images[col, :] = column

  means = np.average(t_images, 1).reshape(784, 1)
  stds = np.std(t_images, 1)
  for col in range(0, t_images.shape[0]):
    column = t_images[col, :]
    if stds[col] != 0.0:
      column = column / stds[col]
    t_images[col, :] = column

  # for graphing
  train_errors = []
  test_errors = []
  eps = []

  alpha = .08
  images_t = images.transpose()

  #convert labels to 10 dimensional vector
  t = np.zeros([60000, 10])
  for i in range(0, 60000):
    t_k = np.zeros(10)
    t_k[labels[i]] = 1
    t[i] = t_k
  t = t.transpose()

  #initialize weights
  W = np.random.rand(784, 10) * 2
  bias = np.random.rand(1,10) * .2
  bias = bias - .001
  W = W - 1

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
    g_S = 1.0 / (1.0 + np.exp(-S))

    #calculate error every 10 epochs
    if e % 10 == 0:
      #training error
      temp = np.add(np.dot(W_t, images), bias.transpose())
      temp = 1/ (1 + np.exp(-temp))
      #output = np.argmax(temp, 0)
      error = np.power(t - temp, 2)
      error = .1 * error.sum()
      error = error/ 60000.0
      train_errors.append(error)
      
      #test error
      t_S = np.add(np.dot(W_t, t_images), bias.transpose())
      t_g_S = 1.0 / (1.0 + np.exp(-t_S))
      t_output = np.argmax(t_g_S, 0)
      error_rate = np.not_equal(t_output, t_labels)
      error_rate = error_rate.sum()
      error_rate = error_rate / 10000.0
      test_errors.append(error_rate)
      eps.append(e)

      print 'error at epoch ', e, 'is: ', error

    #backwards pass
    delta = (g_S - t_batch) * g_S * (1.0 - g_S)
    a = alpha / pow(e + 1, .3)
    #update
    W = W - a * np.dot(images_batch, delta.transpose())
    bias = bias - a * delta.sum(1)

  plt.plot(eps, train_errors, 'g')
  plt.show()
  return W, bias

def train_single_layer_ce(images, labels, t_images, t_labels, epochs):
  """
  train a single layer neural network
  using cross entropy error
  """

  means = np.average(images, 0).reshape(1, 60000)
  images = np.subtract(images, means)
  stds = np.std(images, 0)
  for col in range(0, images.shape[1]):
    column = images[:, col]
    if stds[col] != 0.0:
      column = column / stds[col]
    images[:, col] = column


  means = np.average(t_images, 0).reshape(1, 10000)
  stds = np.std(t_images,0)
  for col in range(0, t_images.shape[1]):
    column = t_images[:, col]
    if stds[col] != 0.0:
      column = column / stds[col]
    t_images[:, col] = column
  # for graphing
  train_errors = []
  test_errors = []
  eps = []

  alpha = .007
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
  bias = np.random.rand(1,10) * .2
  bias = bias - .001
  W = W - .001

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
    g_S = 1.0 / (1.0 + np.exp(-S))

    #calculate error every 10 epochs
    if e % 10 == 0:
      #training error
      temp = np.add(np.dot(W_t, images), bias.transpose())
      y = 1/ (1 + np.exp(-temp))
      temp = t * np.log(y + .00000000000001) + (1.0-t) * np.log(1.0 - y + .00000000000001) # log + small value
      #output = np.argmax(y, 0)
      #error = np.not_equal(output, labels)
      error = -.1 * temp.sum()
      error = error / 60000.0
      train_errors.append(error)
      
      #test error
      t_S = np.add(np.dot(W_t, t_images), bias.transpose())
      t_g_S = 1.0 / (1.0 + np.exp(-t_S))
      t_output = np.argmax(t_g_S, 0)
      error_rate = np.not_equal(t_output, t_labels)
      error_rate = error_rate.sum()
      error_rate = error_rate / 10000.0
      test_errors.append(error_rate)
      eps.append(e)

      print 'error at epoch ', e, 'is: ', error

    #backwards pass
    delta = (g_S - t_batch) 
    a = alpha / pow(e + 1, .38)
    #update
    W = W - a * np.dot(images_batch, delta.transpose())
    bias = bias - a * delta.sum(1)

  plt.plot(eps, train_errors, 'g')
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
  images = images.astype(float)


  labels = labels[0][0]
  labels = labels.flatten()

  test = sio.loadmat('test.mat')
  test = test['test']
  test_images = test['images']
  test_labels = test['labels']
  test_images = test_images[0][0]
  test_images = test_images.reshape(28*28, 10000)
  test_images = test_images.astype(float)


  test_labels = test_labels[0][0]
  test_labels = test_labels.flatten()

  W, bias = train_single_layer_ce(images,labels, test_images, test_labels, 500)
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
