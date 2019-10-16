import os
import re
import json
import itertools

import keras
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

from utils.image import *
from problems import BaseProblem
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
        self.input_shape = dataset['input_shape']

        self.train_generator = None
        self.test_generator = None
        self.data_augmentation = None

        self.verbose = False

        self._initialize_blocks()
        self._generate_configurations()

    def _initialize_blocks(self):
        self.blocks = {
            'input': ['InputLayer', 'batch_input_shape'],
            'conv': ['Conv2D', 'filters', 'kernel_size', 'strides', 'padding', 'activation'],
            'avgpool': ['AveragePooling', 'pool_size', 'strides', 'padding'],
            'maxpool': ['MaxPooling2D', 'pool_size', 'strides', 'padding'],
            'dropout': ['Dropout', 'rate'],
            'upsamp': ['UpSampling2D', 'size'],
            'concat': ['Concatenate', 'axis']
        }

    def _generate_configurations(self):
        kernels = [i[0] for i in self.parser.GRAMMAR['<kernel_size>']]
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

    def _is_valid_config(self, config, img_size):

        return config in self.conv_valid_configs[str(img_size)]

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
            if self._is_valid_config(this_config, img_size):
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
                    print(this_config, '>>>>', new_config)
                    if self._repair_mapping(phenotype, input_shape, index, configurations):
                        return True

            # if all possibilities are invalid or can't be used, this solutions
            # is invalid
            return False
        #elif phenotype[index][0] == 'concat':


        # nothing to be validated, call next block
        return self._repair_mapping(phenotype, input_shape, index+1)

    def _parse_value(self, value):
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
            else:
                return value
        else:
            return value

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
            base_block['config'][key] = self._parse_value(value)

        return base_block

    def _add_layer_to_model(self, model, layer):
        if len(model['config']['layers']) > 0:
            last = model['config']['layers'][-1]['name']
            layer['inbound_nodes'].append([[last, 0, 0]])
        model['config']['layers'].append(layer)
        return model

    def _wrap_up_model(self, model):
        layers = model['config']['layers']
        stack = []
        for i, layer in enumerate(layers[1:]):
            if layer['class_name'] == 'MaxPooling2D':
                stack.append(layers[i])
            elif layer['class_name'] == 'Concatenate':
                other = stack.pop()
                layer['inbound_nodes'][0].insert(0, [other['name'], 0, 0])

        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])

    def _repair_genotype(self, genotype, phenotype):
        values = {}
        model = json.loads(phenotype)
        layers = model['config']['layers']
        for layer in layers:
            print(layer)
            name = layer['name'].split('_')[0]
            if not name in ['conv', 'maxpool', 'avgpool', 'upsamp']:
                continue
            for key in layer['config']:
                vkey = 'kernel_size' if key in ['pool_size', 'size'] else key
                if vkey in values:
                    values[vkey].append(layer['config'][key])
                else:
                    values[vkey] = [layer['config'][key]]

        for key in values:
            #list of values, check the indexes
            grm_options = self.parser.GRAMMAR[f'<{key}>']
            gen_indexes = genotype[self.parser.NT.index(f'<{key}>')]
            fen_indexes = [grm_options.index([val]) for val in values[key]]
            print(key, values[key], grm_options, gen_indexes, fen_indexes)
            #for val in values[key]:
            #     print([val] in grm_options)
            #     fen_val = grm_options.index(val)
            #     gen_val = grm_options[gen]
            #     print(fen_val, gen_val)

    def _map_genotype_to_phenotype(self, genotype):
        
        derivation = self.parser.dsge_recursive_parse(genotype)
        derivation = self._reshape_mapping(derivation)
        print('before', derivation)
        valid = self._repair_mapping(derivation)
        print('after', derivation)

        if not valid:
            return None

        self.naming = {}
        self.stack = []
        model = {
            'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        input_layer = self._build_block('input', [(None,)+self.dataset['input_shape']])
        self._add_layer_to_model(model, input_layer)
        # print(input_layer)

        for i, layer in enumerate(derivation):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            self._add_layer_to_model(model, block)
            # print(block)

        self._wrap_up_model(model)

        #valid = self.repair_json_model(model)

        return json.dumps(model)

    def evaluate(self, solution):

        try:
            json_model = self._map_genotype_to_phenotype(solution.genotype)

            if not json_model:
                return -1, None

            model = model_from_json(json_model)

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
        except Exception as e:
            print(e)
            return -1, None

    def repair_json_model(self, model):
        in_layer = model['config']['input_layers']
        out_layer = model['config']['output_layers']
        layers = model['config']['layers']

        for layer in layers:
            print(layer)

        return True
