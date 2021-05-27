import json
import os
import re
import numpy as np

from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils import np_utils

from cbioge.problems import DNNProblem
from cbioge.problems.dnn import ModelRunner
from cbioge.utils import checkpoint as ckpt

class CNNProblem(DNNProblem):

    def __init__(self, parser, dataset):
        super().__init__(parser, dataset)

        # classification specific
        self.loss = 'categorical_crossentropy'

    def _read_dataset(self, data_dict):
        ''' Reads a dataset stored in dict

            expects a dict with the following keys:
            x_train, y_train
            x_valid, y_valid
            x_test, y_test
            input_shape
            num_classes
        '''

        super()._read_dataset(data_dict)

        # classification specific
        self.num_classes = data_dict['num_classes']
        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

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

        self.naming = {}

        mapping, genotype = self.parser.dsge_recursive_parse(genotype)
        mapping = self._reshape_mapping(mapping)

        mapping.insert(0, ['input', (None,)+self.input_shape]) #input layer
        mapping.append(['dense', self.num_classes, 'softmax']) #output layer

        model = {'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        for i, layer in enumerate(mapping):
            #print(layer)
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            model['config']['layers'].append(block)

        self._wrap_up_model(model)

        return json.dumps(model)

    def evaluate(self, solution):
        super().evaluate(solution)