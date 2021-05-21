import os
import itertools
import json
import numpy as np
import pickle
import re

from math import sin, cos, exp, log

from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, History

from cbioge.utils import checkpoint as ckpt
from cbioge.utils.image import *
from cbioge.utils.model import TimedStopping, plot_acc, plot_loss

from .problem import BaseProblem

class CNNProblem(BaseProblem):

    def __init__(self, parser_):
        self.parser = parser_

        self.batch_size = 10
        self.epochs = 1
        self.training = True
        self.timelimit = 3600

        self.loss = 'categorical_crossentropy'
        self.opt = Adam(lr = 1e-4)
        self.metrics = ['accuracy']

        self.verbose = False

        self._initialize_blocks()
        #self._create_layers_base()
        #self._generate_configurations()

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

        self.x_train = data_dict['x_train']
        self.y_train = data_dict['y_train']
        self.x_valid = data_dict['x_valid']
        self.y_valid = data_dict['y_valid']
        self.x_test = data_dict['x_test']
        self.y_test = data_dict['y_test']
        self.input_shape = data_dict['input_shape']
        self.num_classes = data_dict['num_classes']

        self.train_size = len(self.x_train)
        self.valid_size = len(self.x_valid)
        self.test_size = len(self.x_test)

        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def _initialize_blocks(self):
        self.blocks = {
            'input': ['InputLayer', 'batch_input_shape'],
            'conv': ['Conv2D', 'filters', 'kernel_size', 'strides', 'padding', 'activation'],
            'avgpool': ['AveragePooling2D', 'pool_size', 'strides', 'padding'],
            'maxpool': ['MaxPooling2D', 'pool_size', 'strides', 'padding'],
            'dropout': ['Dropout', 'rate'],
            'dense': ['Dense', 'units', 'activation'],
            'flatten': ['Flatten'],
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
            # if block == 'conv':
            #     end = index+6
            # elif block == 'avgpool' or block == 'maxpool':
            #     end = index+4
            # else:
            #     end = index+2
            end = index + len(self.blocks[block])

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

        if block_name in self.naming:
            self.naming[block_name] += 1
        else:
            self.naming[block_name] = 0
        name = f'{block_name}_{self.naming[block_name]}'

        base_block['class_name'] = self.blocks[block_name][0]
        base_block['name'] = name
        for name, value in zip(self.blocks[block_name][1:], params):
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

    def evaluate(self, phenotype=None, model=None, predict=False, save_model=False):
        try:
            if model is None:
                model = model_from_json(phenotype)
            
            model.compile(loss=self.loss, optimizer=self.opt, metrics=self.metrics)

            x_train = self.x_train[:self.train_size]
            y_train = self.y_train[:self.train_size]
            x_valid = self.x_valid[:self.valid_size]
            y_valid = self.y_valid[:self.valid_size]
            x_test = self.x_test[:self.test_size]
            y_test = self.y_test[:self.test_size]

            ts = TimedStopping(seconds=self.timelimit, verbose=self.verbose)
            callbacks = [ts]

            if save_model:
                mc = ModelCheckpoint(
                    filepath=os.path.join(ckpt.ckpt_folder, f'model_weights.hdf5'), 
                    monitor='val_accuracy', save_weights_only=True, save_best_only=True)
                callbacks.append(mc)

            if self.training:
                history = model.fit(x_train, y_train, 
                    validation_data=(x_valid, y_valid), batch_size=self.batch_size, 
                    epochs=self.epochs, verbose=self.verbose, callbacks=callbacks)
            scores = model.evaluate(x_test, y_test, batch_size=self.batch_size, 
                verbose=self.verbose)

            if self.verbose:
                print('scores', scores)

            if predict:
                if not os.path.exists(ckpt.ckpt_folder):
                    os.mkdir(ckpt.ckpt_folder)

                predictions = model.predict(x_test, batch_size=self.batch_size, 
                    verbose=self.verbose)

                np.save(os.path.join(ckpt.ckpt_folder, 'predictions.npy'), 
                    predictions)

                plot_acc(history)
                plot_loss(history)

                #print(predictions.shape)
                #print(predictions)
                
            return scores, model.count_params()

        except Exception as e:
            print('[evaluation]', e)
            return (1, -1), 0 #(loss, acc), params
