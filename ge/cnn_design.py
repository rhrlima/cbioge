import json
import numpy as np
import sys

from keras import layers
from keras.models import Sequential, model_from_json
from keras.utils import np_utils

import grammar_parser as parser

from keras import backend as K
K.set_image_dim_ordering('th')


class Solution:

	genotype = None
	phenotype = None

	def __init__(self, genes):
		self.genotype = rand.randint(0, 255, rand.randint(1, 10)) \
		if not genes else genes


class CnnProblem:

	input_shape = None
	output_shape = None

	X_train = None
	Y_train = None
	X_test = None
	Y_test = None

	def __init__(self, dataset):
		self.input_shape = (1, 28, 28)
		self.output_shape = 10
		(self.X_train, self.Y_train), (self.X_test, self.Y_test) = dataset.load_data()
		print('train shape', self.X_train.shape, self.Y_train.shape)


	def pre_process(self):

		self.X_train= self.X_train.reshape(self.X_train.shape[0], 1, 28, 28)
		self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 28, 28)

		self.X_train= self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		self.X_train/= 255
		self.X_test /= 255

		self.Y_train = np_utils.to_categorical(self.Y_train, 10)
		self.Y_test = np_utils.to_categorical(self.Y_test, 10)
		print('reshape', self.X_train.shape, self.Y_train.shape)


def create_solution(genes=None):

	return Solution(genes)


from keras.datasets import mnist
p = CnnProblem(mnist)
p.pre_process()
def evaluate(solution):
	
	model = solution.phenotype
	model.compile(
		loss='categorical_crossentropy', 
		optimizer='adam', 
		metrics=['accuracy']
	)
	model.fit(p.X_train, p.Y_train, batch_size=128, epochs=1, verbose=1)
	score = model.evaluate(p.X_test, p.Y_test, verbose=0)
	
	print('Score -> loss: {}\taccuracy: {}'.format(score[0], score[1]))



INPUT_SHAPE = (1, 28, 28)
OUTPUT_SHAPE = 10
def build_model_from_genotype(solution):

	add_input_shape = True
	add_flatten = True
	add_output_shape = True

	print('genotype')
	print(solution.genotype)
	
	deriv = parser.parse(solution.genotype)
	
	print('phenotype')
	print(deriv)

	nodes = []
	node = {'class_name': None, 'config': {}}

	print('parsing')
	index = 0
	while index < len(deriv):

		key, value = deriv[index:index+2]
		
		if key == 'class_name':

			if node[key] is not None:
				nodes.append(node)
				node = {'class_name': None, 'config': {}}

			if add_input_shape:
				node['config']['input_shape'] = INPUT_SHAPE
				add_input_shape = False

			if value == 'Dense' and add_flatten:
				nodes.append({'class_name': 'Flatten', 'config': {}})
				add_flatten = False

			print('{}: {}'.format(key, value))
			node[key] = value
		else:
			print('\t{}: {}'.format(key, value))
			node['config'][key] = eval(value)

		index += 2
	else:
		if add_output_shape:
			node['config']['units'] = OUTPUT_SHAPE
			add_output_shape = False
		nodes.append(node)
		nodes.append(\
			{'class_name': 'Activation', 'config': {'activation': 'softmax'}})

	print('building')
	model = Sequential()
	for n in nodes:
		print(n)
		node = layers.deserialize(n)
		model.add(node)
	solution.phenotype = model



if __name__ == '__main__':
	
	rand = np.random
	parser.load_grammar(sys.argv[1])

	# [92, 176, 129, 232] smallest cnn (Conv+Dense)
	solution = create_solution()
	build_model_from_genotype(solution)
	print(solution.phenotype)
	evaluate(solution)