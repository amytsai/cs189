import scipy.io as sio
import numpy

spam = sio.loadmat("spam.mat")
xtest = spam['Xtest']
xtrain = spam['Xtrain']
ytrain = spam['ytrain']
