import itertools
import json
import numpy as np
import pickle
import re

from math import sin, cos, exp, log

from keras.models import model_from_json
from keras.utils import np_utils

from utils.image import *

from .problem import BaseProblem

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
            self._load_dataset_from_pickle(dataset)
        self._create_layers_base()
        self._generate_configurations()

    def _load_dataset_from_pickle(self, pickle_file):
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

    def _create_layers_base(self):
        self.layers = {
            'input': ['InputLayer', 'batch_input_shape'],
            'conv': ['Conv2D', 'filters', 'kernel_size', 'strides', 'padding', 'activation'],
            'avgpool': ['AveragePooling2D', 'pool_size', 'strides', 'padding'],
            'maxpool': ['MaxPooling2D', 'pool_size', 'strides', 'padding'],
            'dropout': ['Dropout', 'rate'],
            'dense': ['Dense', 'units'],
        }

    def _generate_configurations(self):
        kernels = [i[0] for i in self.parser.GRAMMAR['<ksize>']]
        strides = [i[0] for i in self.parser.GRAMMAR['<strides>']]
        padding = [i[0] for i in self.parser.GRAMMAR['<padding>']]
        conv_configs = list(itertools.product(kernels, strides, padding))
        max_img_size = self.input_shape[1]
        self.conv_valid_configs = {}
        for img_size in range(0, max_img_size+1):
            key = str(img_size)
            self.conv_valid_configs[key] = conv_configs[:] #copies the configs list
            for config in conv_configs:
                if calculate_output_size((img_size, img_size), *config) <= (0, 0):
                    self.conv_valid_configs[key].remove(config)

    def _reshape_mapping(self, phenotype):

        new_mapping = []

        index = 0
        while index < len(phenotype):
            block = phenotype[index]
            if block == 'conv':
                end = index+6
            elif block == 'avgpool' or block == 'maxpool':
                end = index+4
            else:
                end = index+2

            new_mapping.append(phenotype[index:end])
            phenotype = phenotype[end:]

        return new_mapping

    def _repair_mapping(self, phenotype, input_shape=None, index=0, configurations=None):

        #print('#'*index, index)

        # if the mapping reached the end, without problems, return TRUE
        if index >= len(phenotype):
            return True

        input_shape = self.input_shape if input_shape is None else input_shape
        img_size = input_shape[1]

        # the repair occurs just for convolution or pooling
        if phenotype[index][0] in ['conv', 'maxpool', 'avgpool']:

            # get the needed parameters for each type of block (convolution or pooling)    
            if phenotype[index][0] == 'conv':
                start, end = 2, 5
            if phenotype[index][0] in ['maxpool', 'avgpool']:
                start, end = 1, 4

            this_config = tuple(phenotype[index][start:end])

            # if the current config is VALID, calculate output and call next block
            if this_config in self.conv_valid_configs[str(img_size)]:
                output_shape = calculate_output_size(input_shape, *this_config)
                #print(this_config, 'is valid', input_shape, output_shape)
                return self._repair_mapping(phenotype, output_shape, index+1)
            else:

                # if the current config is not VALID, generate a list of indexes 
                # of the possible configurations and shuffles it
                if configurations is None:
                    configurations = np.arange(len(self.conv_valid_configs[str(img_size)]))
                    np.random.shuffle(configurations)

                # if the current config is in the possibilities but can't be used
                # remove the index corresponding to it
                if this_config in self.conv_valid_configs[str(img_size)]:
                    cfg_index = self.conv_valid_configs[str(img_size)].index(this_config)
                    configurations.remove(cfg_index)

                # for each new config, try it by calling the repair to it
                for cfg_index in configurations:
                    new_config = self.conv_valid_configs[str(img_size)][cfg_index]
                    phenotype[index][start:end] = list(new_config)
                    if self._repair_mapping(phenotype, input_shape, index, configurations):
                        return True

            # if all possibilities are invalid or can't be used, this solutions
            # is invalid
            return False

        # nothing to be validated, call next block
        return self._repair_mapping(phenotype, input_shape, index+1)

    def _parse_value(self, value):
        #value = value.replace(' ', '')
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

    def _build_block(self, block_name, params):

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        if block_name in self.names:
            self.names[block_name] += 1
        else:
            self.names[block_name] = 0
        name = f'{block_name}_{self.names[block_name]}'

        base_block['class_name'] = self.layers[block_name][0]
        base_block['name'] = name
        for name, value in zip(self.layers[block_name][1:], params):
            base_block['config'][name] = self._parse_value(value)
        #print(base_block)
        return base_block

    def _add_layer_to_model(self, model, layer):
        if len(model['config']['layers']) > 0:
            last = model['config']['layers'][-1]['name']
            layer['inbound_nodes'].append([[last, 0, 0]])
        model['config']['layers'].append(layer)

    def _wrap_up_model(self, model):
        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])

    def _map_genotype_to_phenotype(self, genotype):

        derivation = self.parser.dsge_recursive_parse(genotype)
        derivation = self._reshape_mapping(derivation)
        valid = self._repair_mapping(derivation)

        if not valid:
            return None

        self.names = {}
        model = {
            'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        input_layer = self._build_block('input', [self.input_shape])

        self._add_layer_to_model(model, input_layer)

        for i, layer in enumerate(derivation):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            self._add_layer_to_model(model, block)

        model = self._wrap_up_model(model)

        return json.dumps(model)

    def evaluate(self, solution, verbose=0):

        try:
            json_model = self._map_genotype_to_phenotype(solution.genotype)
            
            if not json_model:
                return -1, None

            model = model_from_json(json_model)

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
