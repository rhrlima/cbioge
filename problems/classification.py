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

    def __init__(self, parser_, dataset=None):
        self.parser = parser_

        self.batch_size = 128
        self.epochs = 1
        self.training = True

        self.loss = 'categorical_crossentropy'
        self.opt = 'adam'
        self.metrics = ['accuracy']

        self.verbose = False

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

            self.train_size = len(self.x_train)
            self.valid_size = len(self.x_valid)
            self.test_size = len(self.x_test)
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

    def _get_layer_outputs(self, mapping):
        outputs = []
        depth = 0
        for i, block in enumerate(mapping):
            name, params = block[0], block[1:]
            if name == 'input':
                output_shape = self.input_shape
            elif name == 'conv':
                output_shape = calculate_output_size(output_shape, *params[1:4])
                output_shape += (params[0],)
            elif name in ['maxpool', 'avgpool']:
                temp = calculate_output_size(output_shape, *params[:3])
                output_shape = temp + (output_shape[2],)
            print('\t'*depth, i, output_shape, block)
            outputs.append(output_shape)
        return outputs

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

    def _wrap_up_model(self, model):
        layers = model['config']['layers']
        for i, layer in enumerate(layers[1:]):
            last = model['config']['layers'][i]
            layer['inbound_nodes'].append([[last['name'], 0, 0]])

        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])

    def map_genotype_to_phenotype(self, genotype):

        self.names = {}

        mapping = self.parser.dsge_recursive_parse(genotype)
        mapping = self._reshape_mapping(mapping)
        
        #repaired = self._repair_mapping(mapping)

        # if not repaired:
        #     return None

        # model = {'class_name': 'Model', 
        #     'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        # mapping.insert(0, ['input', (None,)+self.input_shape])
        for i, layer in enumerate(mapping):
            print(layer)
        #     block_name, params = layer[0], layer[1:]
        #     block = self._build_block(block_name, params)
        #     model['config']['layers'].append(block)

        # self._wrap_up_model(model)

        # return json.dumps(model)
        return None

    def evaluate(self, phenotype):
        try:
            model = model_from_json(phenotype)
            
            model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metrics)

            x_train = self.x_train[:self.train_size]
            y_train = self.y_train[:self.train_size]
            x_valid = self.x_valid[:self.valid_size]
            y_valid = self.y_valid[:self.valid_size]
            x_test = self.x_test[:self.test_size]
            y_test = self.y_test[:self.test_size]

            if self.training:
                model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
            scores = model.evaluate(x_valid, y_valid, verbose=self.verbose)

            if self.verbose:
                print('scores', scores)

            return scores
        except Exception as e:
            print('[evaluation]', e)
            return -1
