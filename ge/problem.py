import pickle

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

	intput_shape = None
	output_shape = None

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


	def map_genotype_to_phenotype(self, genotype):
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


	def evaluate(self, phenotype):
		#super(CnnProblem, self).evaluate(phenotype)
		print('Cnn Problem implementation')


if __name__ == '__main__':
	p = CnnProblem()

	pickle_file = '../datasets/mnist/mnist.pickle'
	p.load_dataset_from_pickle(pickle_file)