import numpy as np

from keras import layers
from keras.models import Sequential

import grammar_parser as parser

OUTPUT_SHAPE = 10

class Solution:

	genotype = None
	phenotype = None

	def __init__(self):
		self.genotype = rand.randint(0, 255, rand.randint(1, 10))


def create_solution():

	return Solution()


def evaluate(solution):
	pass


def build_from_genotype(solution):

	print(solution.genotype)
	temp = parser.parse(solution.genotype)
	pass


import json
if __name__ == '__main__':
	
	rand = np.random
	parser.load_grammar('cnn2.bnf')

	solution = create_solution()
	build_from_genotype(solution)

	config = {
		'class_name': 'Dense', 
		'config': {
			'name': 'dense_0', 
			'units': 10, 
		}
	}

	d = layers.deserialize(config)
	print(d.get_config())