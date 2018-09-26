import json
import numpy as np
import sys

from keras import layers
from keras.models import Sequential

import grammar_parser as parser

INPUT_SHAPE = (1, 28, 28)
INPUT_DIM = len(INPUT_SHAPE)-1
OUTPUT_SHAPE = 10


class Solution:

	genotype = None
	phenotype = None

	def __init__(self, genes):
		self.genotype = rand.randint(0, 255, rand.randint(1, 10)) \
		if not genes else genes


def create_solution(genes=None):

	return Solution(genes)


def evaluate(solution):
	pass


def build_from_genotype(solution): #['Conv', '64', '(3,3)', 'Dense', '32']

	print('genotype')
	print(solution.genotype)
	
	deriv = parser.parse(solution.genotype)
	
	print('phenotype')
	print(deriv)

	nodes = []
	node = {'config': {}}

	print('parsing')
	index = 0
	while index < len(deriv):

		key, value = deriv[index:index+2]
		
		if key == 'class_name':

			if key in node:
				nodes.append(node)
				node = {'config': {}}

			print('{}: {}'.format(key, value))
			node[key] = value
		else:
			print('\t{}: {}'.format(key, value))
			node['config'][key] = eval(value)

		index += 2
	else:
		nodes.append(node)

	print('building')
	for n in nodes:
		print(n)
		print(layers.deserialize(n))


if __name__ == '__main__':
	
	rand = np.random
	parser.load_grammar(sys.argv[1])

	solution = create_solution([92, 176, 129, 232])
	build_from_genotype(solution)