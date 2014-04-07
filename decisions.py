import scipy
from collections import Counter
from math import *
import numpy as np

def entropy(binarized_examples):
	""" calculates the entropy of a list of binarized examples
		returns a floating point number """
	X = binarized_examples
	if len(X) == 0:
		assert False
		return 0.0
	P_x1 = sum(X) / float(len(X)) # probability X is true
	if P_x1 == 1.0 or P_x1 == 0.0:
		return 0.0
	P_x0 = 1.0 - P_x1 # probability X is false
	assert P_x1 < 1
	assert P_x1 > 0
	result = - P_x1 * log(P_x1, 2)
	result -= P_x0 * log(P_x0, 2)

	return result

def binarize(examples_X, attribute, split):
	""" returns boolean array of whether attribute is > split """
	return examples_X[: , attribute] > split

def infoGain(examples_X, examples_Y, attribute, split):
	H_Y = entropy(examples_Y.flatten())
	X = binarize(examples_X, attribute, split)
	P_x1 = sum(X)/float(len(X)) # probability X is true
	if sum(X) == 0 or sum(X) == len(X):
		return 0.0
	assert P_x1 != 1.0 and P_x1 != 0.0
	P_x0 = 1 - P_x1 # probability X is false
	H_Y1 = entropy(examples_Y.flatten()[X==1]) #entropy Y | X = 1
	H_Y0 = entropy(examples_Y.flatten()[X==0]) #entropy Y | X = 0

	return H_Y - P_x0 * H_Y0 - P_x1 * H_Y1

def optimal_split(examples_X, examples_Y, attribute):
	""" Find the optimal split point of an attribute given examples and
		labels

		For the attribute, sort values and calculate infoGain continuing to increment
		until the infoGain stops improving
	"""
	X = examples_X[:, attribute]
	X = np.sort(X) #sorted attributes
	if sum(X) == 0 or sum(X) == 0.0:
		return 0.0, 0.0
	last_I= 0.0;
	last_split = float("-inf")
	for i in range(0, len(X)):
		split = X[np.argmax(X > last_split)] #next largest number
		I = infoGain(examples_X, examples_Y, attribute, split)
		#print "split: ", split, "infogain: ", I
		if I < last_I: # infoGain did not improve
			return (split + last_split) / 2, last_I #return halfway between current split and last split
		else:
			last_I = I
			last_split = split

	second_largest = X[np.argmax(X > 0.0)]
	return (0.0 + second_largest)/2, last_I

def split_attribute(examples_X, examples_Y):
	""" returns a list containing
		[0]: split attribute index
		[1]: the value to split at
		if there is 0 information gain from splitting on any of the attributes, return -1"""
	assert not sum(examples_Y.flatten()) == 0
	assert not sum(examples_Y.flatten()) == len(examples_Y.flatten())
	numAttributes = examples_X.shape[1]
	infoGains = np.zeros(numAttributes)
	splits = np.zeros(numAttributes)
	for attr in range(0, numAttributes):
		#print "trying attr: " ,attr
		# calcuate information gain of splitting at each attribute optimally
		split, gain = optimal_split(examples_X, examples_Y, attr)
		#print "optimal split for attr " , attr  , " is " , split
		splits[attr] = split
		infoGains[attr] = gain

	if(sum(infoGains) == 0.0): # no infogain from splitting anymore
		return (-1, -1)
	else:
		result = np.argmax(infoGains)
		return (result, splits[result])


class DecisionTree:

	def __init__(self, attribute = None, split = None, extractor = lambda x: x, left = None, right = None):
		self.attribute = attribute
		self.split = split
		self.extractor = extractor
		self.left = left
		self.right = right

	def __str__(self):
		if not self.is_leaf:
			return '<DecisionTree with %u child nodes>' % (self.left + self.right)
		else:
			return '<DecisionTree leaf with leaf value "' + str(self.leaf_value) + '">'

	def is_leaf(self):
		return self.left is None and self.right is None

	def check_valid(self):
		return self.leaf_value is not None or (self.nodes and len(self.nodes) > 1) # either we can make a choice or we're a leaf

	def _greater_than_split(self, features):
		assert not self.is_leaf()
		feature = features[self.attribute]
		return feature > self.split

	def choose(self, obj):
		cur_node = self
		if not self.is_leaf():
			features = self.extractor(obj)
		while not cur_node.is_leaf():
			if cur_node._greater_than_split(features):
				cur_node = cur_node.right
			else:
				cur_node = cur_node.left
		return cur_node.split

	@staticmethod
	def leaf(leaf_value):
		return DecisionTree(split = leaf_value)

	# determines if an object will get to a specific node in this DecisionTree
	## BROKEN after structure change
	def hits_node(self, node, obj):
		cur_node = self
		if cur_node is node:
			return True
		while cur_node.leaf_value is None:
			index = self.fn(obj)
			if index:
				if index is True:
					index = 1
				cur_node = cur_node.nodes[index]
			else:
				cur_node = cur_node.nodes[0]
			if cur_node is node:
				return True
		return False

	# takes a list of possible objects, and returns only those that encounter a specific node
	def filter_by_node(self, node, list_of_obj):
		return filter(lambda x: self.hits_node(node, x), list_of_obj)

leaf = DecisionTree.leaf

def grow_tree(examples_X, examples_Y, depth = 0):
	"""
	function that actually builds the decision tree. yay
	"""
	print "DEPTH = %u" % depth

	if(sum(examples_Y.flatten()) == 0): # if all labels are 0
		#print "I AM A LEAF :DDD"
		return leaf(0)
	elif(sum(examples_Y.flatten()) == len(examples_Y.flatten())): # if all labels are 1
		print "I AM A 1 LEAF"
		return leaf(1)
	else:
		#print "Current entropy = %f" % entropy(examples_Y.flatten())
		#print "looking for optimal split attribute"
		sa = split_attribute(examples_X, examples_Y);
		attribute = sa[0]
		split = sa[1]
		if(attribute != -1 and split != -1):
			#print "attribute = " + str(attribute)
			#print "split = " + str(split)
			indices = binarize(examples_X, attribute, split)
			Set1X = examples_X[indices]
			Set1Y = examples_Y[indices]
			indices = np.invert(indices)
			Set0X  = examples_X[indices]
			Set0Y  = examples_Y[indices]

			return DecisionTree(attribute, split, lambda x: x, grow_tree(Set0X, Set0Y, depth + 1), grow_tree(Set1X, Set1Y, depth + 1))
		else:
			P = sum(examples_Y.flatten())/len(examples_Y.flatten())
			print "Can't perform any more splits; P = ", P
			if P > .5:
				return leaf(1)
			else:
				return leaf(0)

class RandomForest:

	def __init__(self, trees):
		self.trees = trees

	def __str__(self):
		return "<RandomForest of %u trees>" % len(self.trees)

	def choose(self, obj):
		answers = Counter()
		for tree in self.trees:
			answers[tree.choose(obj)] += 1
		return max(answers, key = lambda x: answers[x])


def sanity_check():
	def isStanford(s):
		if s == "Stanford":
			return True
		return False
	def isStanfordNumerical(s):
		if s == "Stanford":
			return 1
		return 0

	blueleaf = DecisionTree.leaf("blue")
	redleaf = DecisionTree.leaf("red")
	schoolcolor = DecisionTree(isStanford, (blueleaf, redleaf))
	nschoolcolor = DecisionTree(isStanfordNumerical, (blueleaf, redleaf))
	forest = RandomForest((schoolcolor, nschoolcolor))

	assert schoolcolor.choose("Cal") == "blue"
	assert nschoolcolor.choose("Cal") == "blue"
	assert schoolcolor.choose("Stanford") == "red"
	assert nschoolcolor.choose("Stanford") == "red"
	assert nschoolcolor.choose("Everyone") == "blue"

	assert forest.choose("Cal") == "blue"
	assert forest.choose("Stanford") == "red"

	assert not schoolcolor.hits_node(blueleaf, "Stanford")
	assert schoolcolor.hits_node(blueleaf, "Cal")
	a = ["Cal", "Stanford", "Harvard", "MIT"]
	assert schoolcolor.filter_by_node(blueleaf, a) == ["Cal", "Harvard", "MIT"]
	assert schoolcolor.filter_by_node(redleaf, a) == ["Stanford"]

s = sanity_check

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
