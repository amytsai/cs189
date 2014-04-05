import scipy
from collections import Counter
from math import *
import numpy as np

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
	""" Find the optimal split point of an attribute given examples and
		labels

		For the attribute, sort values and calculate infoGain continuing to increment
		until the infoGain stops improving
	"""
	X = examples_X[:, attribute]
	X = np.sort(X)
	print X
	last_I= 0;
	last_split = float("-inf")
	for i in range(0, len(X)):
		split = X[np.argmax(X > last_split)]
		I = infoGain(examples_X, examples_Y, attribute, split)
		print "split: ", split, "infogain: ", I
		if (I - last_I) < 0: # infoGain did not improve
			return split
		else:
			last_I = I
			last_split = split

	return 0.0

def split_attribute(examples_X, examples_Y):
	""" returns a list containing
		[0]: split attribute index
		[1]: the value to split at """
	numAttributes = examples_X.shape[1]
	infoGains = np.zeros(numAttributes)
	splits = np.zeros(numAttributes)
	for attr in range(0, numAttributes):
		print "trying attr: " ,attr
		# calcuate information gain of splitting at each attribute optimally
		split = optimal_split(examples_X, examples_Y, attr)
		print "optimal split for attr " , attr  , " is " , split
		splits[attr] = split
		infoGains[attr] = infoGain(examples_X, examples_Y, attr, split)

	result = np.argmax(infoGains)
	return (result, splits[result])

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

	return H_Y - P_x0 * H_Y0 - P_x1 * H_Y1

class DecisionTree:

	def __init__(self, split, left = None, right = None):
		self.split = split
		self.left = left
		self.right = right

	def __str__(self):
		if self.nodes: # if there's something in nodes
			return '<DecisionTree with %u child nodes>' % len(self.nodes)
		else:
			return '<DecisionTree leaf with leaf value "' + str(self.leaf_value) + '">'

	def check_valid(self):
		return self.leaf_value is not None or (self.nodes and len(self.nodes) > 1) # either we can make a choice or we're a leaf

	def choose(self, obj):
		cur_node = self
		while cur_node.leaf_value is None:
			index = self.fn(obj)
			if index:
				if index is True:
					index = 1
				cur_node = cur_node.nodes[index]
			else:
				cur_node = cur_node.nodes[0]
		return cur_node.leaf_value

	@staticmethod
	def leaf(leaf_value):
		return DecisionTree(None, None, leaf_value)

def grow_tree(examples_X, examples_Y):
	"""
	function that actually builds the decision tree. yay
	"""
	if(sum(examples_Y.flatten()) == 0): # if all labels are 0
		return DecisionTree(0)
	elif(sum(examples_Y.flatten()) == len(examples_Y.flatten())): # if all labels are 1
		return DecisionTree(1)
	else:
		print "looking for optimal split attribute"
		sa = split_attribute(examples_X, examples_Y);
		attribute = sa[0]
		split = sa[1]
		print "attribute = " + attribute
		print "split = " + split
		indices = binarize(examples_X, attribute, split)
		Set1X = examples_X[indices]
		Set1Y = examples_Y[indices]
		indices = np.invert(indices)
		Set0X  = examples_X[indices]
		Set0Y  = examples_Y[indices]

		return DecisionTree(split, grow_tree(Set0X, Set0Y), grow_tree(Set1X, Set1Y))

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
	blueleaf = DecisionTree.leaf("blue")
	redleaf = DecisionTree.leaf("red")
	print blueleaf
	print redleaf

	def isStanford(s):
		if s == "Stanford":
			return True
		return False

	def isStanfordNumerical(s):
		if s == "Stanford":
			return 1
		return 0

	schoolcolor = DecisionTree(isStanford, (blueleaf, redleaf))
	nschoolcolor = DecisionTree(isStanfordNumerical, (blueleaf, redleaf))
	print schoolcolor
	print nschoolcolor
	forest = RandomForest((schoolcolor, nschoolcolor))
	print forest

	print "Cal is " + schoolcolor.choose("Cal")
	print "Cal is " + nschoolcolor.choose("Cal")
	print "Stanford is " + schoolcolor.choose("Stanford")
	print "Stanford is " + nschoolcolor.choose("Stanford")
	print "Everyone else is also " + nschoolcolor.choose("Everyone")

	print "The forest thinks that Cal is " + forest.choose("Cal")
	print "The forest thinks that Stanford is " + forest.choose("Stanford")

s = sanity_check