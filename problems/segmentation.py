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
            'conv': ['Conv2D', 'filters', 'kernel_size', 'strides', 'padding', 'activation'],
            'avgpool': ['AveragePolling', 'pool_size', 'padding'],
            'maxpool': ['MaxPolling', 'pool_size', 'padding'],
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
                end = index+3
            else:
                end = index+2

            new_mapping.append(phenotype[index:end])
            phenotype = phenotype[end:]

        return new_mapping

    def _build_block(self, block):
        block_name, params = block[0], block[1:]

        if block_name in self.naming:
            self.naming[block_name] += 1
        else:
            self.naming[block_name] = 0
        name = f'{block_name}_{self.naming[block_name]}'

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        base_block['class_name'] = self.blocks[block_name][0]
        base_block['name'] = name
        for key, value in zip(self.blocks[block_name][1:], params):
            base_block['config'][key] = value

        print(base_block)
        return base_block

    def map_genotype_to_phenotype(self, genotype):
        self.naming = {}

        print(genotype)

        phenotype = self.parser.dsge_recursive_parse(genotype)
        phenotype = self._reshape_mapping(phenotype)

        print(phenotype)
        for block in phenotype:
            json_block = self._build_block(block)

        return phenotype

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