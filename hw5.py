import scipy
import scipy.io as sio
import numpy as np
import code
from decisions import *

leaf = DecisionTree.leaf # for convenience
def train(data, labels)
	return grow_tree(data,labels)
	
def predict(data, labels, DT):
	error = 0
	predictions = np.zeros(data.shape[0])
	for ex in range(0, data.shape[0]):
		prediction = DT.choose(data[ex])
		predictions[ex] = prediction
		if(prediction != labels[ex, 0]):
			error += 1
	error_rate = error/ data.shape[0]
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
	print "test error rate = ",  error/ xtrain.shape[0]

	#print DT
	code.interact(local = locals())


if __name__ == "__main__":
    main()
