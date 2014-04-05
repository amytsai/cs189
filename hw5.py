import scipy
import scipy.io as sio
import numpy as np
from decisions import *

leaf = DecisionTree.leaf # for convenience

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
	print DT


if __name__ == "__main__":
    main()

	


