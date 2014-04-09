import scipy
import scipy.io as sio
import numpy as np
import code
import signal
from decisions import *
import random as rand
from multiprocessing import Pool

#KFOLD = 4

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
	N = arg[2]
	m = arg[3]
	i = arg[4]
	subset = sample(range(0, data.shape[0]), N)
	tree = grow_rand_tree(data[subset], labels[subset], m)
	print "generated tree #%u" % i
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
	iterable = [(data, labels, N, m, i) for i in xrange(T)]
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
	print "test error rate = ", error_rate
	return predictions

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

def main():
	"""
	just some testing stuff for now
	"""

	spam = sio.loadmat("spam.mat")
	xtest = spam['Xtest']
	xtrain = spam['Xtrain']
	ytrain = spam['ytrain']

	# forest = train_rand_forest(xtrain, ytrain, 10, 400, 15)
	# predict_rand_forest(xtrain, ytrain, forest)
	# 7.2%:
	# train = lambda x, y: train_rand_forest(x, y, 15, 500, 25)
	# 6.3%:
	# train = lambda x, y: train_rand_forest(x, y, 15, 500, 25)
	# 6.1%:
	# train = lambda x, y: train_rand_forest(x, y, 15, 600, 20)
	# 7.5%:
	# train = lambda x, y: train_rand_forest(x, y, 15, 400, 20)
	# 6.2%:
	# train = lambda x, y: train_rand_forest(x, y, 21, 800, 20)
	# 6.9%:
	# train = lambda x, y: train_rand_forest(x, y, 21, 400, 20)
	# 6.5%:
	# train = lambda x, y: train_rand_forest(x, y, 21, 600, 20)
	# 6.1% 
	# train = lambda x, y: train_rand_forest(x, y, 21, 900, 20)
	# 5.8%
	# train = lambda x, y: train_rand_forest(x, y, 21, 1100, 20)
	train = lambda x, y: train_rand_forest(x, y, 21, 1500, 20)
	predict = lambda x, tree: tree.choose(x)
	cross_validate(10, xtrain, ytrain, train, predict)

	"""
	OLD STUFF
	DT = train(xtrain, ytrain)
	error = predict(xtrain,ytrain, DT)
	print "training error rate = ",  error/ float(xtrain.shape[0])

	print "Beginning %u-fold cross-validation..." % KFOLD
	random_classes = np.random.random_integers(0, KFOLD - 1, len(xtrain))
	cross_trees = []
	errors = []
	for i in xrange(KFOLD):
		print "Cross-validation iteration %u" % i
		train_indices = (random_classes != i).nonzero()
		test_indices = (random_classes == i).nonzero() # apparently boolean advanced indexing is faster than this, but meh :(
		train_x = xtrain[train_indices]
		train_y = ytrain[train_indices]
		test_x = xtrain[test_indices]
		test_y = ytrain[test_indices]
		print "Growing tree..."
		tree = train(train_x, train_y)
		print "Tree grown. "
		error = 0
		for i in xrange(len(test_x)):
			if tree.choose(test_x[i]) != test_y[i]:
				error += 1
		error_rate = float(error) / float(len(test_x))

		print "Error rate: " + str(error_rate)
		errors.append(error_rate)
		cross_trees.append(tree)

	print "Total average error rate: " + str(sum(errors) / len(errors))
	"""




if __name__ == "__main__":
	main()
