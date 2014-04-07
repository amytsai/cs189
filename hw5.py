import scipy
import scipy.io as sio
import numpy as np
import code
from decisions import *
import random as rand

#KFOLD = 4

leaf = DecisionTree.leaf # for convenience
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

def train_rand_forest(data, labels, T, N ,m ):
	"""
	Train random forest
	T: number of tree
	N: number of samples per tree
	m: number of attributes sampled per node
	"""
	trees = []
	for t in range(0, T):
		subset = rand.sample(range(0, data.shape[0]), N)
		tree = grow_rand_tree(data[subset], labels[subset], m)
		trees.append(tree)
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
		error_rate = predict_fn(test_x, test_y, tree)
		print "Error rate: " + str(error_rate)
		errors.append(error_rate)
		cross_trees.append(tree)

	print "Total average error rate: " + str(sum(errors) / len(errors))

def main():
	"""
	just some testing stuff for now
	"""

	spam = sio.loadmat("spam.mat")
	xtest = spam['Xtest']
	xtrain = spam['Xtrain']
	ytrain = spam['ytrain']

	forest = train_rand_forest(xtrain, ytrain, 10, 400, 15)
	predict_rand_forest(xtrain, ytrain, forest)
	#cross_validate(4, xtrain, ytrain)

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

	code.interact(local = locals())



if __name__ == "__main__":
    main()
