import scipy

class DecisionTree:

	def __init__(self, fn, true_node, false_node, leaf_value = None):
		self.fn = fn
		self.true_node = true_node
		self.false_node = false_node
		self.leaf_value = leaf_value

	def recursive_choice(self, obj):
		if fn(obj):
			return self.true_node.choice(obj)
		else:
			return self.false_node.choice(obj)

	def choice(self, obj):
		cur_node = self
		while cur_node.leaf_value is not None:
			if fn(obj):
				cur_node = cur_node.left
			else:
				cur_node = cur_node.right
		return cur_node.leaf_value
