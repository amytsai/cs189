import scipy
import scipy.io as sio
import numpy as np
import code
import signal
from decisions import *
import random as rand
import csv
from multiprocessing import Pool
from math import *

KFOLD = 4

def train(data, labels):
	"""train a regular decision tree"""
	return grow_tree(data,labels)

def predict(data, labels, DT):
	"""predict using a regular decision tree"""
	error = 0
	predictions = np.zeros(data.shape[0])
	for ex in xrange(0, data.shape[0]):
		prediction = DT.choose(data[ex])
		predictions[ex] = prediction
		if(prediction != labels[ex, 0]):
			error += 1
	error_rate = error/ float(data.shape[0])
	print "test error rate = ", error_rate
	return predictions

def sample(*args):
	return rand.sample(*args)

def _train_rand_tree(arg):
	data = arg[0]
	labels = arg[1]
	T = arg[2]
	N = arg[3]
	m = arg[4]
	i = arg[5]
	subset = sample(range(0, data.shape[0]), N)
	tree = grow_rand_tree(data[subset], labels[subset], m)
	print "Finished generating tree " + str(i + 1) + " of " + str(T)
	return tree

def init_worker():
	signal.signal(signal.SIGINT, signal.SIG_IGN)

def train_rand_forest(data, labels, T, N ,m ):
	"""
	Train random forest
	T: number of tree
	N: number of samples per tree
	m: number of attributes sampled per node
	"""
	pool = Pool(None, init_worker)
	iterable = [(data, labels, T, N, m, i) for i in xrange(T)]
	try:
		trees = pool.map(_train_rand_tree, iterable)
	except KeyboardInterrupt:
		pool.terminate()
		sys.exit(1)
	pool.close()
	return RandomForest(trees)

def predict_rand_forest(data, labels, forest):
	error = 0
	predictions = np.zeros(data.shape[0])
	for ex in xrange(0, data.shape[0]):
		prediction = forest.choose(data[ex])
		predictions[ex] = prediction
		if(prediction != labels[ex, 0]):
			error += 1
	error_rate = error/ float(data.shape[0])
	#print "test error rate = ", error_rate
	return predictions

def train_boosted(data, labels, T, m, max_depth = 2):
	"""
	adaboost
	using the same algorithm as "IntroToBoosting.pdf"
	"""
	h = []
	a = np.zeros(T)
	N = data.shape[0]
	D = np.zeros([T,N])
	D[0, :]= np.tile(1/float(N), N)
	for t in range(0, T):
		#resample data
		print "iteration: " , t
		sums  = np.zeros(N)
		sums[0]= 0
		#cumulative sums
		for i in range(1, N):
		 	sums[i] = D[t,i-1] + sums[i-1]
		data_t = np.zeros(data.shape)
		labels_t = np.zeros(labels.shape)
		for n in range(0, N):
			i = rand.random()
			sample_index = np.argmax(sums > i)
			data_t[n] = data[sample_index]
			labels_t[n] = labels[sample_index]

		#train weak learner
		weak_learn = grow_pruned_tree(data_t, labels_t, max_depth, m)
		h.append(weak_learn)

		#calculate a_t
		error = 0
		for i in range(N):
			if weak_learn.choose(data_t[i]) != labels_t[i]:
				error += 1
		error_rate = float(error) / float(N)
		assert error_rate != 0
		print "error_rate = ", error_rate
		a[t] = .5 * log((1-error_rate)/error_rate)

		#update
		if(t != T-1):
			for i in range(0, N):
				if labels_t[i] == weak_learn.choose(data_t[i]):
					D[t+1, i] = exp(-a[t])
				else:
					D[t+1, i] = exp(a[t])
			D[t+1, :] = D[t+1, :]/float(np.sum(D[t+1, :]))

	return a, h

def predict_boosted(a, h, data, labels):
	prediction = np.zeros(data.shape[0])
	for i in range(0,data.shape[0]):
		x = data[i, :]
		H = 0
		for t in range(len(a)):
			y = h[t].choose(x)
			if y == 0:
				y = -1
			H += y * a[t]
		prediction[i] = np.sign(H)
	prediction = prediction == 1 #convert -1's to 0
	error = 0
	for i in range(0, data.shape[0]):
		if prediction[i] != labels[i]:
			error += 1
	error_rate = error/float(data.shape[0])
	print "error rate = ", error_rate
	return prediction

def cross_validate(k, data, labels, train_fn = train, predict_fn = predict):
	print "Beginning %u-fold cross-validation..." % k
	random_classes = np.random.random_integers(0, k - 1, len(data))
	cross_trees = []
	errors = []
	for i in xrange(k):
		print "Cross-validation iteration %u" % i
		train_indices = (random_classes != i).nonzero()
		test_indices = (random_classes == i).nonzero() # apparently boolean advanced indexing is faster than this, but meh :(
		train_x = data[train_indices]
		train_y = labels[train_indices]
		test_x = data[test_indices]
		test_y = labels[test_indices]
		print "Growing tree..."
		tree = train_fn(train_x, train_y)
		print "Tree grown. "
		error = 0
		for i in xrange(len(test_x)):
			if predict_fn(test_x[i], tree) != test_y[i]:
				error += 1
		error_rate = float(error) / float(len(test_x))
		print "Error rate: " + str(error_rate)
		errors.append(error_rate)
		cross_trees.append(tree)

	print "Total average error rate: " + str(sum(errors) / len(errors))
	return errors, cross_trees

def generate_predictions(xtrain, ytrain, xtest):
	forest = train_rand_forest(xtrain, ytrain, 51, 1100, 15)
	predictions = np.array(map(forest.choose, xtest))
	return predictions

def main():

	spam = sio.loadmat("spam.mat")
	xtest = spam['Xtest']
	xtrain = spam['Xtrain']
	ytrain = spam['ytrain']

	# generate predictions and win
	print "Starting Random Forest generation..."
	predictions = generate_predictions(xtrain, ytrain, xtest)
	print "Forest generation complete, writing to file..."
	with open('hw5.csv', 'w') as f:
		f.write('Id,Category\n')
		for index, item in enumerate(predictions):
			f.write(str(index + 1) + ',' + str(item) + '\n')
	print "File written, see hw5.csv. "

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
