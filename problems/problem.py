# import sys; sys.path.append('..')

import json
import numpy as np
import pickle
import re

from math import sin, cos, exp, log

from keras.models import model_from_json
from keras.utils import np_utils


DEBUG = False


class BaseProblem:

    def map_genotype_to_phenotype(self, solution):
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, phenotype):
        raise NotImplementedError('Not implemented yet.')


class CNNProblem(BaseProblem):

    parser = None

    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    x_test = None
    y_test = None

    input_shape = None
    num_classes = None

    batch_size = 128
    epochs = 1

    loss = 'categorical_crossentropy'
    opt = 'adam'
    metrics = ['accuracy']

    def __init__(self, parser_, dataset=None):
        self.parser = parser_
        if dataset:
            self.load_dataset_from_pickle(dataset)

        self._create_layers_base()
        

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

        deriv = self.parser.parse(genotype)

        if not deriv:
            return None

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
            if DEBUG:
                print(n)
            model['config'].append(n)

        # returns the model as string
        return json.dumps(model)

    def _create_layers_base(self):
        self.names = {}
        self.layers = {
            'input': ['InputLayer', 'batch_input_shape:'],
            'conv': ['Conv2D', 'filters:int', 'kernel_size:int', 'strides:int', 'padding:', 'activation:'],
            'avgpool': ['AveragePooling2D', 'pool_size:int', 'padding:'],
            'dense': ['Dense', 'units:int'],
        }

    def _reshape_mapping(self, phenotype):

        new_mapping = []

        index = 0
        while index < len(phenotype):
            if phenotype[index] == 'conv':
                end = index+6
            elif phenotype[index] == 'avgpool':
                end = index+3
            else:
                end = 2

            new_mapping.append(phenotype[index:end])
            phenotype = phenotype[end:]

        return new_mapping

    def _build_block(self, block_name, params):

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        if block_name in self.names:
            self.names[block_name] += 1
        else:
            self.names[block_name] = 0
        name = f'{block_name}_{self.names[block_name]}'

        base_block['class_name'] = self.layers[block_name].pop(0)
        base_block['name'] = name
        for name, value in zip(self.layers[block_name], params):
            name, op = name.split(':')
            if op != '' :
                value = int(value) if op == 'int' else float(value)
            base_block['config'][name] = value

        return base_block

    def _add_layer_to_model(self, model, layer):
        if len(model['config']['layers']) > 0:
            last = model['config']['layers'][-1]['name']
            layer['inbound_nodes'].append([[last, 0, 0]])
        model['config']['layers'].append(layer)
        return model

    def _wrap_up_model(self, model):
        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])
        return model

    def map_v2(self, genotype):

        deriv = self.parser.dsge_recursive_parse(genotype)

        print(deriv)
        deriv = self._reshape_mapping(deriv)
        print(deriv)
        
        model = {'class_name': 'Model', 'config': {'layers': [], 'input_layers': [], 'output_layers': []}, }

        input_layer = self._build_block('input', [self.input_shape])

        self._add_layer_to_model(model, input_layer)

        for i, layer in enumerate(deriv):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            self._add_layer_to_model(model, block)

        model = self._wrap_up_model(model)

        print(model)

        return json.dumps(model)

    def evaluate(self, solution, verbose=0):

        try:
            json_model = self.map_genotype_to_phenotype(solution.genotype)
            model = model_from_json(json_model)

            if not model:
                return -1, None

            model.compile(
                loss=self.loss,
                optimizer=self.opt,
                metrics=self.metrics)

            # train
            if verbose:
                print('[training]')
            model.fit(self.x_train, self.y_train, batch_size=self.batch_size,
                      epochs=self.epochs, verbose=verbose)

            # valid
            if verbose:
                print('[validation]')
            score = model.evaluate(self.x_valid, self.y_valid, verbose=verbose)

            if verbose:
                print('loss: {}\taccuracy: {}'.format(score[0], score[1]))

            return score[1], json_model

        except Exception as e:
            if DEBUG:
                print(e)
                print('[evaluation] invalid model from solution: {}'.format(
                    solution.genotype))
            return -1, None


class DNNProblem(BaseProblem):

    parser = None

    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    x_test = None
    y_test = None

    input_shape = None
    num_classes = None

    batch_size = 128
    epochs = 1

    loss = 'categorical_crossentropy'
    opt = 'adam'
    metrics = ['accuracy']

    def __init__(self, parser_, dataset=None):
        self.parser = parser_
        if dataset:
            self.load_dataset_from_pickle(dataset)

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

        deriv = self.parser.dsge_parse(genotype)

        if not deriv:
            return None

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
            if DEBUG:
                print(n)
            model['config'].append(n)

        # returns the model as string
        return json.dumps(model)
