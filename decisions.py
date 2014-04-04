import scipy

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

	def choice(self, obj):
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
	def make_leaf(leaf_value):
		return DecisionTree(None, None, leaf_value)



def sanity_check():
	blueleaf = DecisionTree.make_leaf("blue")
	redleaf = DecisionTree.make_leaf("red")
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

	print "Cal is " + schoolcolor.choice("Cal")
	print "Cal is " + nschoolcolor.choice("Cal")
	print "Stanford is " + schoolcolor.choice("Stanford")
	print "Stanford is " + nschoolcolor.choice("Stanford")
	print "Everyone else is also " + nschoolcolor.choice("Everyone")

