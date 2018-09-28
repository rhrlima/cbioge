import grammar as parser
import pickle

from keras import layers
from keras.models import Sequential, model_from_json
from keras.utils import np_utils

#label first
from keras import backend as K
K.set_image_dim_ordering('th')


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

	def load_dataset_from_pickle(self, pickle_file):
		print('Loading the dataset')
		with open(pickle_file, 'rb') as f:
			temp = pickle.load(f)
			
			self.x_train = temp['train_dataset']
			self.y_train = temp['train_labels']

			self.x_valid = temp['valid_dataset']
			self.y_valid = temp['valid_labels']

			self.x_test = temp['test_dataset']
			self.y_test = temp['test_labels']

			del temp
			print('Training set', self.x_train.shape, self.y_train.shape)
			print('Validation set', self.x_valid.shape, self.y_valid.shape)
			print('Test set', self.x_test.shape, self.y_test.shape)

		self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, 28, 28)
		self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, 28, 28)

		self.y_train = np_utils.to_categorical(self.y_train, 10)
		self.y_test = np_utils.to_categorical(self.y_test, 10)


	def map_genotype_to_phenotype(self, solution):
		add_input_shape = True
		add_flatten = True
		add_output_shape = True

		print('genotype')
		print(solution.genotype)
		
		deriv = parser.parse(solution.genotype)
		if not deriv: return
		
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
					node['config']['input_shape'] = self.input_shape
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
				node['config']['units'] = self.num_classes
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


	def evaluate(self, solution):

		self.map_genotype_to_phenotype(solution)

		if not solution.phenotype:
			print('invalid solution')
			return

		model = solution.phenotype
		model.compile(
			loss='categorical_crossentropy', 
			optimizer='adam', 
			metrics=['accuracy']
		)
		model.fit(p.x_train, p.y_train, batch_size=128, epochs=1, verbose=1)
		score = model.evaluate(p.x_test, p.y_test, verbose=0)
		
		print('Score -> loss: {}\taccuracy: {}'.format(score[0], score[1]))
		

class Solution:

	genotype = None
	phenotype = None

	def __init__(self, genes=None):
		self.genotype = rand.randint(0, 255, rand.randint(1, 10)) \
		if not genes else genes


def test():
	return parser.grammar


if __name__ == '__main__':
	p = CnnProblem()

	parser.load_grammar('cnn2.bnf')

	pickle_file = '../datasets/mnist/mnist.pickle'
	p.load_dataset_from_pickle(pickle_file)
	p.input_shape = (1, 28, 28)
	p.num_classes = 10
	
	import numpy as np
	rand = np.random
	solution = Solution()
	print(solution.genotype)

	p.evaluate(solution)