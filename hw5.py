import scipy
import scipy.io as sio
import numpy as np
import code
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
	code.interact(local = locals())


if __name__ == "__main__":
    main()

	


