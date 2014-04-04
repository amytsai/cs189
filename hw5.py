import scipy
import scipy.io as sio
import numpy as np
from decisions import *

spam = sio.loadmat("spam.mat")
xtest = spam['Xtest']
xtrain = spam['Xtrain']
ytrain = spam['ytrain']

leaf = DecisionTree.leaf # for convenience

def entropy(binarized_examples):
	""" calculates the entropy of a list of binarized examples 
		returns a floating point number """
	X = binarized_examples
	result = 0
	P_x1 = sum(X)/float(len(X)) # probability X is true
	P_x0 = 1 - P_x1 # probability X is false

	result -= P_x1 * log(P_x1, 2) 
	result -= P_x0 * log(P_x0, 2) 

	return result

def optimal_split(examples_X, examples_Y, attribute):
	""" TODO: find the optimal split point of an attribute given examples and
		labels """
	return 0.0


def binarize(examples_X, attribute, split):
	""" returns boolean array of whether attribute is > split """
	return examples_X[: , attribute] > split


def infoGain(examples_X, examples_Y, attribute, split):
	H_Y = entropy(examples_Y.flatten())
	X = binarize(examples_X, attribute, split)

    P_x1 = sum(X)/float(len(X)) # probability X is true
	P_x0 = 1 - P_x1 # probability X is false
	H_Y1 = entropy(examples_Y.flatten()[X==1]) #entropy Y | X = 1
	H_Y0 = entropy(examples_Y.flatten()[X==0]) #entropy Y | X = 0

	return H_Y - P_x0 * H_Y0 + P_x1 * H_Y1

	


