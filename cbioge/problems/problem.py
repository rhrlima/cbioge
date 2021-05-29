import os
import re
import numpy as np

from keras.models import model_from_json
from keras.optimizers import Adam

from cbioge.problems.dnn import ModelRunner
from cbioge.utils import checkpoint as ckpt


class BaseProblem:
    ''' This BaseProblem class should be used as a reference of what a 
        problem class must have to be used with the evolutionary algorithms. 
        During the evolution the methods in this class are called.
    '''

    def map_genotype_to_phenotype(self, genotype) -> str:
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, solution) -> bool:
        raise NotImplementedError('Not implemented yet.')


class DNNProblem(BaseProblem):
    ''' Base class used for Problems related to the automatic design of
        deep neural networks. Specific behavior must be implemented in child
        classes after calling the super() funcions.
    '''

    def __init__(self, parser, dataset):
        self.parser = parser
        self._read_dataset(dataset)
        self.blocks = self.parser.blocks

        self.batch_size = 10
        self.epochs = 1
        self.timelimit = None

        self.opt = Adam(lr = 1e-4)
        self.metrics = ['accuracy']

        self.workers = 1
        self.multiprocessing = False

        self.verbose = False

    def _read_dataset(self, data_dict):
        ''' Reads a dataset stored in dict

            expects a dict with the following keys:
            x_train, y_train
            x_valid, y_valid
            x_test, y_test
            input_shape
            num_classes
        '''
        if data_dict is None:
            raise AttributeError('data_dict cannot be NoneType')

        self.x_train = data_dict['x_train']
        self.y_train = data_dict['y_train']
        self.x_valid = data_dict['x_valid']
        self.y_valid = data_dict['y_valid']
        self.x_test = data_dict['x_test']
        self.y_test = data_dict['y_test']
        self.input_shape = data_dict['input_shape']

        self.train_size = len(self.x_train)
        self.valid_size = len(self.x_valid)
        self.test_size = len(self.x_test)

    def _reshape_mapping(self, mapping):
        # groups layer name and parameters together
        new_mapping = []
        index = 0
        while index < len(mapping):
            block = mapping[index]
            end = index + len(self.blocks[block])
            new_mapping.append(mapping[index:end])
            mapping = mapping[end:]
        return new_mapping

    def _parse_value(self, value):
        # TODO buscar maneira melhor de representar rand(min, max) na gramatica
        ''' parses a string in the form of "[int, int]" or "[float, float]"
            to the correct types and return a random between the interval
        '''
        if type(value) is str:
            m = re.match('\\[(\\d+[.\\d+]*),\\s*(\\d+[.\\d+]*)\\]', value)
            if m:
                min_ = eval(m.group(1))
                max_ = eval(m.group(2))
                if type(min_) == int and type(max_) == int:
                    return np.random.randint(min_, max_)
                elif type(min_) == float and type(max_) == float:
                    return np.random.uniform(min_, max_)
                else:
                    raise TypeError('type mismatch')
        return value

    def _base_build(self, mapping):

        self.naming = {}

        model = {'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        for i, layer in enumerate(mapping):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            model['config']['layers'].append(block)

        return model

    def _build_block(self, block_name, params):

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        if block_name in self.naming:
            self.naming[block_name] += 1
        else:
            self.naming[block_name] = 0
        name = f'{block_name}_{self.naming[block_name]}'

        base_block['class_name'] = self.blocks[block_name][0]
        base_block['name'] = name
        for name, value in zip(self.blocks[block_name][1:], params):
            base_block['config'][name] = self._parse_value(value)
        return base_block

    def _wrap_up_model(self, model):
        # iterates over layers and add previous layer as input of current one
        for i, layer in enumerate(model['config']['layers'][1:]):
            last = model['config']['layers'][i]
            layer['inbound_nodes'].append([[last['name'], 0, 0]])

        # creates and adds input and output layers to model
        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])

    def evaluate(self, solution):
        ''' Evaluates the phenotype

            phenotype: json structure containing the network architecture
            weights: network weights (optional)

        '''
        try:
            model = model_from_json(solution.phenotype)
            model.compile(
                loss=self.loss, 
                optimizer=self.opt, 
                metrics=self.metrics)

            # defines the portion of the dataset being used for 
            # training, validation, and test
            x_train = self.x_train[:self.train_size]
            y_train = self.y_train[:self.train_size]
            x_valid = self.x_valid[:self.valid_size]
            y_valid = self.y_valid[:self.valid_size]
            x_test = self.x_test[:self.test_size]
            y_test = self.y_test[:self.test_size]

            solution_path = os.path.join(ckpt.ckpt_folder, f'solution_{solution.id}')
            runner = ModelRunner(model, path=solution_path, verbose=self.verbose)
            runner.train_model(x_train, y_train, 
                self.batch_size, 
                self.epochs, 
                validation_data=(x_valid, y_valid), 
                timelimit=self.timelimit)
            runner.test_model(x_test, y_test, 
                self.batch_size)

            # local changes for checkpoint
            solution.fitness = runner.accuracy
            solution.params = runner.params
            solution.evaluated = True

            return True

        except Exception as e:
            print('[evaluation]', e)
            solution.fitness = -1
            solution.params = 0
            solution.evaluated = True

            return False
