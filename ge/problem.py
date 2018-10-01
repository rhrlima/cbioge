import grammar as parser
import pickle
import json

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

		self.x_train = self.x_train.reshape(
			(self.x_train.shape[0],)+self.input_shape)
		self.x_valid = self.x_valid.reshape(
			(self.x_valid.shape[0],)+self.input_shape)
		self.x_test = self.x_test.reshape(
			(self.x_test.shape[0],)+self.input_shape)

		self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
		self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
		self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)


	def map_genotype_to_phenotype(self, solution):
		add_input_shape = True
		add_flatten = True
		add_output_shape = True
		
		deriv = parser.parse(solution.genotype)
		if not deriv: return

		nodes = []
		node = {'class_name': None, 'config': {}}

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

				node[key] = value
			else:
				node['config'][key] = eval(value)

			index += 2
		else:
			if add_output_shape:
				node['config']['units'] = self.num_classes
				add_output_shape = False
			nodes.append(node)
			nodes.append(
				{'class_name': 'Activation', 'config': {'activation': 'softmax'}})

		model = {'class_name': 'Sequential', 'config': []}
		for n in nodes: 
			model['config'].append(n)
		solution.phenotype = model_from_json(json.dumps(model))


	def evaluate(self, solution, verbose=0):

		self.map_genotype_to_phenotype(solution)

		if not solution.phenotype: return -1

		model = solution.phenotype
		
		model.compile(
			loss='categorical_crossentropy', 
			optimizer='adam', 
			metrics=['accuracy']
		)
		model.fit(
			self.x_train, 
			self.y_train, 
			batch_size=128, 
			epochs=1, 
			verbose=verbose
		)

		score = model.evaluate(self.x_valid, self.y_valid, verbose=verbose)
		
		print('Score -> loss: {}\taccuracy: {}'.format(score[0], score[1]))
		return score[1]


if __name__ == '__main__':

	print('testing mode')