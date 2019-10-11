import os

import keras
from keras.preprocessing.image import ImageDataGenerator

from .problem import BaseProblem
from datasets.dataset import DataGenerator


class UNetProblem(BaseProblem):

    parser = None

    batch_size = 1
    epochs = 1

    loss = 'binary_crossentropy'
    opt = 'Adam'
    metrics = ['accuracy']

    def __init__(self, parser, dataset):
        self.parser = parser
        self.dataset = dataset

        self.train_generator = None
        self.test_generator = None
        self.data_augmentation = None

        self.verbose = False

        self._initialize_blocks()

    def _initialize_blocks(self):
        self.blocks = {
            'input': ['InputLayer', 'batch_input_shape'],
            'conv': ['Conv2D', 'filters', 'kernel_size', 'strides', 'padding', 'activation'],
            'avgpool': ['AveragePolling', 'pool_size', 'strides', 'padding'],
            'maxpool': ['MaxPolling', 'pool_size', 'strides', 'padding'],
            'dropout': ['Dropout', 'rate'],
            'upsamp': ['UpSampling2D', 'size'],
            'concat': ['Concatenate', 'axis']
        }

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

    def _build_block(self, block_name, params):

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        if block_name in self.naming:
            self.naming[block_name] += 1
        else:
            self.naming[block_name] = 0
        name = f'{block_name}_{self.naming[block_name]}'

        base_block['class_name'] = self.blocks[block_name][0]
        base_block['name'] = name
        for key, value in zip(self.blocks[block_name][1:], params):
            base_block['config'][key] = value

        return base_block

    def _map_genotype_to_phenotype(self, genotype):
        
        derivation = self.parser.dsge_recursive_parse(genotype)
        derivation = self._reshape_mapping(derivation)
        # valid = self._repair_mapping(derivation)

        self.naming = {}
        model = {
            'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        input_layer = self._build_block('input', [self.dataset['input_shape']])
        print(input_layer)
        # self._add_layer_to_model(model, input_layer)

        for i, layer in enumerate(derivation):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            print(block)
            # self._add_layer_to_model(model, block)

        # model = self._wrap_up_model(model)

        return {}# json.dumps(model)

    def evaluate(self, solution):

        model = solution.phenotype
        if not model:
            return -1

        model.compile(
            optimizer=self.opt, 
            loss=self.loss, 
            metrics=self.metrics)

        model.fit_generator(
            self.train_generator, 
            self.dataset['train_steps'], 
            self.epochs, verbose=self.verbose)

        loss, acc = model.evaluate_generator(
            self.test_generator, 
            self.dataset['test_steps'], verbose=self.verbose)

        return acc