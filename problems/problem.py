import sys
sys.path.append('..')

from grammars import grammar as parser
from algorithms import ge
import json
import numpy as np
import pickle
import re

from keras import layers
from keras.models import Sequential, model_from_json
from keras.utils import np_utils

#label first
from keras import backend as K
#K.set_image_dim_ordering('th')


DEBUG = False


class BaseProblem:

	def map_genotype_to_phenotype(self, solution):
		raise NotImplementedError('Not implemented yet.')

	def evaluate(self, phenotype):
		raise NotImplementedError('Not implemented yet.')


class CnnProblem(BaseProblem):

	x_train= None
	y_train= None
	x_valid= None
	y_valid= None
	x_test = None
	y_test = None

	input_shape = None
	num_classes = None

	batch_size = 128
	epochs = 1


	def load_dataset_from_pickle(self, pickle_file):
		with open(pickle_file, 'rb') as f:
			temp = pickle.load(f)
			
			self.x_train = temp['train_dataset']
			self.y_train = temp['train_labels']

			self.x_valid = temp['valid_dataset']
			self.y_valid = temp['valid_labels']

			self.x_test = temp['test_dataset']
			self.y_test = temp['test_labels']

			self.input_shape = temp['input_shape']
			self.num_classes = temp['num_classes']

			del temp

		self.x_train = self.x_train.reshape((-1,)+self.input_shape)
		self.x_valid = self.x_valid.reshape((-1,)+self.input_shape)
		self.x_test = self.x_test.reshape((-1,)+self.input_shape)

		self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
		self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
		self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)


	def map_genotype_to_phenotype(self, genotype):
		add_input_shape = True
		add_flatten = True
		add_output_shape = True
		
		deriv = parser.parse(genotype)

		if not deriv: return None

		nodes = []
		node = {'class_name': None, 'config': {}}

		index = 0
		while index < len(deriv):

			key, value = deriv[index:index+2]
			
			if key == 'class_name':

				if node[key] is not None:
					nodes.append(node)
					node = {'class_name': None, 'config': {}}

				# first Conv node needs input_shape parameter
				if add_input_shape:
					node['config']['input_shape'] = self.input_shape
					add_input_shape = False

				# first Dense node needs Flatten before
				if value == 'Dense' and add_flatten:
					nodes.append({'class_name': 'Flatten', 'config': {}})
					add_flatten = False

				node[key] = value
			else:
				# range pattern
				m = re.match('\\[(\\d+[.\\d+]*),\\s*(\\d+[.\\d+]*)\\]', value)
				if m:
					min_ = eval(m.group(1))
					max_ = eval(m.group(2))
					if type(min_) == int and type(max_) == int:
						value = np.random.randint(min_, max_)
					elif type(min_) == float and type(max_) == float:
						value = np.random.uniform(min_, max_)
					else:
						raise TypeError('type mismatch')
				else:
					# kernel size pattern
					m1 = re.match('\\((\\d+),\\s*(\\d+)\\)', value)
					m2 = re.match('^\\d+$', value)
					if m1 or m2:
						value = eval(value)

				node['config'][key] = value

			index += 2
		else:
			# last node needs output_shape as number of classes
			# and softmax activation
			if add_output_shape:
				node['config']['units'] = self.num_classes
				node['config']['activation'] = 'softmax'
				add_output_shape = False
			nodes.append(node)

		model = {'class_name': 'Sequential', 'config': []}
		for n in nodes: 
			if DEBUG: print(n)
			model['config'].append(n)

		try:
			model = model_from_json(json.dumps(model))
		except Exception:
			if DEBUG: print('[mapping] invalid model from solution: {}'.format(
				solution.genotype))
			return None

		return model


	def evaluate(self, solution, verbose=0):

		model = self.map_genotype_to_phenotype(solution.genotype)

		if not model: return -1

		solution.phenotype = model
		
		model.compile(
			loss='categorical_crossentropy', 
			optimizer='adam', 
			metrics=['accuracy']
		)
		model.fit(
			self.x_train, 
			self.y_train, 
			batch_size=self.batch_size, 
			epochs=self.epochs, 
			verbose=verbose
		)

		score = model.evaluate(self.x_valid, self.y_valid, verbose=verbose)
		
		# max the accuracy (score[1]) min the loss (score[0])
		fitness = score[1] - score[0]
		
		solution.data['loss'] = score[0]
		solution.data['acc']= score[1]

		if verbose == 1: print('loss: {}\taccuracy: {}'.format(
			score[0], 
			score[1]))
		return fitness


if __name__ == '__main__':

	DEBUG = False
	print('testing mode')

	import numpy as np

	parser.load_grammar('../grammars/cnn.bnf')

	p = CnnProblem()
	p.load_dataset_from_pickle('../datasets/mnist/mnist.pickle')

	while True:
		genes = np.random.randint(0, 255, np.random.randint(1, 10))
		solution = ge.Solution(genes)

		print(genes)
		model = p.map_genotype_to_phenotype(genes)
		solution.phenotype = model

		if model:
			#model.summary()
			p.evaluate(solution, 1)
		#else:
		#	print('None')