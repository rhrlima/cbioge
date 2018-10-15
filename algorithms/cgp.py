import numpy as np

NUM_INPUTS = 1
NUM_OUTPUTS = 1
NUM_NODES = 2
MAX_ARITY = 1

num_funcs = 2


class Node:

	inputs = []
	function = None
	active = True
	output = None


class Solution:

	nodes = []
	output_nodes = []


def create_solution():

	for i in range(NUM_NODES):
		in_v = np.random.randint(0, NUM_INPUTS + i)
		func = np.random.randint(0, num_funcs)
		#node_output = 0
		print(in_v, func)

	for i in range(NUM_OUTPUTS):
		out_v = np.random.randint(0, NUM_INPUTS + NUM_NODES)
		print(out_v)


if __name__ == '__main__':
	create_solution()