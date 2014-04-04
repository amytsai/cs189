import scipy
from collections import Counter

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

class DecisionTree:

	def __init__(self, fn, nodes = None, leaf_value = None):
		self.fn = fn
		self.nodes = nodes
		self.leaf_value = leaf_value

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