import scipy
import scipy.io as sio
import numpy as np
import code
from decisions import *

KFOLD = 4

leaf = DecisionTree.leaf # for convenience
def train(data, labels):
	return grow_tree(data,labels)

def predict(data, labels, DT):
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

def main():
	"""
	just some testing stuff for now
	"""

	spam = sio.loadmat("spam.mat")
	xtest = spam['Xtest']
	xtrain = spam['Xtrain']
	ytrain = spam['ytrain']

	print "entropy of ytrain: "
	print entropy(ytrain.flatten())
	DT = grow_tree(xtrain, ytrain)
	error = 0.0
	for ex in range(0, xtrain.shape[0]):
		prediction = DT.choose(xtrain[ex])
		if(prediction != ytrain[ex, 0]):
			error += 1
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
		tree = grow_tree(train_x, train_y)
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

	code.interact(local = locals())



if __name__ == "__main__":
    main()
