import scipy
import scipy.io as sio
import numpy as np
from decisions import *

spam = sio.loadmat("spam.mat")
xtest = spam['Xtest']
xtrain = spam['Xtrain']
ytrain = spam['ytrain']

leaf = DecisionTree.leaf # for convenience
