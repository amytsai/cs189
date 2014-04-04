import scipy
from collections import Counter

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

	# determines if an object will get to a specific node in this DecisionTree
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

	assert not schoolcolor.hits_node(blueleaf, "Stanford")
	assert schoolcolor.hits_node(blueleaf, "Cal")
	a = ["Cal", "Stanford", "Harvard", "MIT"]
	assert schoolcolor.filter_by_node(blueleaf, a) == ["Cal", "Harvard", "MIT"]
	assert schoolcolor.filter_by_node(redleaf, a) == ["Stanford"]

s = sanity_check
