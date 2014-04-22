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

def train_multilayer_ms(images, labels, t_images, t_labels, epochs):
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
  train_loss = []
  test_errors = []
  eps = []

  alpha = .05
  images_t = images.transpose()

  #convert labels to 10 dimensional vector
  t = np.zeros([60000, 10])
  for i in range(0, 60000):
    t_k = np.zeros(10)
    t_k[labels[i]] = 1
    t[i] = t_k
  t = t.transpose()

  #initialize weights and biases
  W1 = np.random.rand(784, 300) * 2 - 1
  W2 = np.random.rand(300, 100) * 2 - 1
  W3 = np.random.rand(100, 10) * 2 - 1
  bias1 = np.random.rand(1,300) * .2 - .1
  bias2 = np.random.rand(1, 100) * .2 - .1
  bias3 = np.random.rand(1, 10) * .2 - .1

  for e in range(0, epochs):
    # take a mini batch
    sample = np.arange(60000)
    np.random.shuffle(sample)
    sample = sample[0:200]
    t_batch = t[:, sample]
    images_batch = images[:, sample]
    
    #forward pass
    W1_t = W1.transpose()
    W2_t = W2.transpose()
    W3_t = W3.transpose()
    S1 = np.add(np.dot(W1_t, images_batch),bias1.transpose()) #S1 300 x 200
    out1 = np.tanh(S1)
    out1_p = 1.0 - np.power(np.tanh(S1), 2)
    S2 = np.add(np.dot(W2_t, out1), bias2.transpose()) #S2 200 x 100
    out2 = np.tanh(S2)
    out2_p = 1.0 - np.power(np.tanh(S2), 2)
    S3 = np.add(np.dot(W3_t, out2), bias3.transpose())
    out3 = 1.0 / (1.0 + np.exp(-S3))

    #assert False

    #backwards pass
    a = alpha / pow(e + 1, .3)
    delta3 = (out3 - t_batch) * out3 * (1.0 - out3)
    W3 = W3 - a * np.dot(S2, delta3.transpose())
    delta2 = out2_p * np.dot(W3, delta3)
    W2 = W2 - a * np.dot(S1, delta2.transpose())
    delta1 = out1_p * np.dot(W2, delta2)
    W1 = W1 - a * np.dot(images_batch, delta1.transpose())

    bias1 = bias1 - a * delta1.sum(1)
    bias2 = bias2 - a * delta2.sum(1)
    bias3 = bias3 - a * delta3.sum(1)

    #finite differences
    '''
    epsilon = .00001
    out1 = predict_multilayer_me(images_batch, W1 + epsilon, W2, W3, bias1, bias2, bias3)
    out2 = predict_multilayer_me(images_batch, W1 - epsilon, W2, W3, bias1, bias2, bias3)
    E1 = np.power(t_batch - out1, 2).sum()
    E2 = np.power(t_batch - out2, 2).sum()
    fd_delta1 = (E1 - E2)/ 2* epsilon
    '''

    #calculate error every 10 epochs
    if e % 10 == 0:
      #training error
      output = predict_multilayer_me(images, W1, W2, W3, bias1, bias2, bias3)
      y = np.argmax(output, 0)
      error = np.not_equal(y, labels)
      error = error.sum() / 60000.0
      loss = np.power(t - output, 2)
      loss = .1 * error.sum()
      loss = loss/ 60000.0
      train_errors.append(error)
      train_loss.append(loss)
      
      #test error
      t_output = predict_multilayer_me(t_images, W1, W2, W3, bias1, bias2, bias3)
      t_output = np.argmax(t_output, 0)
      error_rate = np.not_equal(t_output, t_labels)
      error_rate = error_rate.sum() / 10000.0
      test_errors.append(error_rate)
      eps.append(e)

      print 'training error at epoch ', e, 'is: ', error
      print 'test error at epoch ', e, 'is: ', error_rate


  plt.plot(eps, train_errors, 'g', eps, train_loss, 'r', eps, test_errors, 'b')
  plt.show()
  return W, bias

def train_multilayer_ce(images, labels, t_images, t_labels, epochs):
  print 'max images: ', images.max()
  images = images / 255.0
  print 'max images: ', images.max()
  means = np.average(images, 1).reshape(784, 1)
  images = np.subtract(images, means)
  stds = np.std(images, 1)
  for col in range(0, images.shape[0]):
    column = images[col, :]
    if stds[col] != 0.0:
      column = column / stds[col]
    images[col, :] = column

  t_images = t_images/ 255.0
  means = np.average(t_images, 1).reshape(784, 1)
  stds = np.std(t_images, 1)
  for col in range(0, t_images.shape[0]):
    column = t_images[col, :]
    if stds[col] != 0.0:
      column = column / stds[col]
    t_images[col, :] = column

  # for graphing
  train_errors = []
  train_loss = []
  test_errors = []
  eps = []

  alpha = .005
  images_t = images.transpose()

  #convert labels to 10 dimensional vector
  t = np.zeros([60000, 10])
  for i in range(0, 60000):
    t_k = np.zeros(10)
    t_k[labels[i]] = 1
    t[i] = t_k
  t = t.transpose()

  #initialize weights and biases
  W1 = np.random.rand(784, 300) * .00002 - .00001
  W2 = np.random.rand(300, 100) * .00002 - .00001
  W3 = np.random.rand(100, 10) * .00002 - .00001
  bias1 = np.random.rand(1,300) * .2 - .1
  bias2 = np.random.rand(1, 100) * .2 - .1
  bias3 = np.random.rand(1, 10) * .2 - .1

  for e in range(0, epochs):
    # take a mini batch
    sample = np.arange(60000)
    np.random.shuffle(sample)
    sample = sample[0:200]
    t_batch = t[:, sample]
    images_batch = images[:, sample]
    
    #forward pass
    W1_t = W1.transpose()
    W2_t = W2.transpose()
    W3_t = W3.transpose()
    S1 = np.add(np.dot(W1_t, images_batch),bias1.transpose()) #S1 300 x 200
    out1 = np.tanh(S1)
    out1_p = 1 - np.power(np.tanh(S1), 2)
    S2 = np.add(np.dot(W2_t, out1), bias2.transpose()) #S2 200 x 100
    out2 = np.tanh(S2)
    out2_p = 1 - np.power(np.tanh(S2), 2)
    S3 = np.add(np.dot(W3_t, out2), bias3.transpose())
    out3 = 1.0 / (1.0 + np.exp(-S3))

    #assert False

    #backwards pass
    a = alpha / pow(e + 1, .5)
    delta3 = (out3 - t_batch)
    W3 = W3 - a * np.dot(S2, delta3.transpose())
    delta2 = out2_p * np.dot(W3, delta3)
    W2 = W2 - a * np.dot(S1, delta2.transpose())
    delta1 = out1_p * np.dot(W2, delta2)
    W1 = W1 - a * np.dot(images_batch, delta1.transpose())

    bias1 = bias1 - a * delta1.sum(1)
    bias2 = bias2 - a * delta2.sum(1)
    bias3 = bias3 - a * delta3.sum(1)

    #finite differences
    '''
    epsilon = .00001
    out1 = predict_multilayer_me(images_batch, W1 + epsilon, W2, W3, bias1, bias2, bias3)
    out2 = predict_multilayer_me(images_batch, W1 - epsilon, W2, W3, bias1, bias2, bias3)
    E1 = np.power(t_batch - out1, 2).sum()
    E2 = np.power(t_batch - out2, 2).sum()
    fd_delta1 = (E1 - E2)/ 2* epsilon
    '''

    #calculate error every 10 epochs
    if e % 10 == 0:
      #training error
      output = predict_multilayer_me(images, W1, W2, W3, bias1, bias2, bias3)
      y = np.argmax(output, 0)
      error = np.not_equal(y, labels)
      error = error.sum() / 60000.0
      loss = t * np.log(output + .00000000000001) + (1.0-t) * np.log(1.0 - output + .00000000000001) # log + small value
      loss = .1 * error.sum()
      loss = loss/ 60000.0
      train_errors.append(error)
      train_loss.append(loss)
      
      #test error
      t_output = predict_multilayer_me(t_images, W1, W2, W3, bias1, bias2, bias3)
      t_output = np.argmax(t_output, 0)
      error_rate = np.not_equal(t_output, t_labels)
      error_rate = error_rate.sum() / 10000.0
      test_errors.append(error_rate)
      eps.append(e)

      print 'training error at epoch ', e, 'is: ', error
      print 'test error at epoch ', e, 'is: ', error_rate


  plt.plot(eps, train_errors, 'g', eps, train_loss, 'r', eps, test_errors, 'b')
  plt.show()
  return W, bias

def predict_multilayer_me(images, W1, W2, W3, b1, b2, b3):
  W1_t = W1.transpose()
  W2_t = W2.transpose()
  W3_t = W3.transpose()
  S1 = np.add(np.dot(W1_t, images),b1.transpose()) #S1 300 x 200
  out1 = np.tanh(S1)
  out1_p = 1 - np.power(np.tanh(S1), 2)
  S2 = np.add(np.dot(W2_t, out1), b2.transpose()) #S2 200 x 100
  out2 = np.tanh(S2)
  out2_p = 1 - np.power(np.tanh(S2), 2)
  S3 = np.add(np.dot(W3_t, out2), b3.transpose())
  out3 = 1.0 / (1.0 + np.exp(-S3))
  return out3

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

  W, bias = train_multilayer_ce(images,labels, test_images, test_labels, 100)
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
