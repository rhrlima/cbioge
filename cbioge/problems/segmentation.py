from cbioge.problems.problem import DNNProblem
import os
import re
import copy
import json
import pickle

from keras.optimizers import Adam
from keras.models import model_from_json
from keras.optimizers import *
from keras.callbacks import *

from cbioge.problems import DNNProblem

from cbioge.utils import checkpoint as ckpt
from cbioge.utils.image import *
from cbioge.utils.model import TimedStopping


class UNetProblem(DNNProblem):

    def __init__(self, parser, dataset):
        super().__init__(parser, dataset)

        # segmentation specific
        self.loss = 'binary_crossentropy'

    def read_dataset_from_generator(self, dataset, train_gen, test_gen):
        self.dataset = dataset
        self.train_generator = train_gen
        self.test_generator = test_gen
        self.input_shape = tuple(dataset['input_shape'])

    def _wrap_up_model(self, model):
        layers = model['config']['layers']
        stack = []
        for i, layer in enumerate(model['config']['layers']):
            if layer['class_name'] in ['push', 'bridge']: #CHECK
                stack.append(layers[i-1]) #layer before (conv)
                model['config']['layers'].remove(layers[i])

        for i, layer in enumerate(layers[1:]):

            last = model['config']['layers'][i]
            layer['inbound_nodes'].append([[last['name'], 0, 0]])

            if layer['class_name'] == 'Concatenate':
                other = stack.pop()
                # print('CONCATENATE', layer['name'], other['name'])
                layer['inbound_nodes'][0].insert(0, [other['name'], 0, 0])

        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])

    def _repair_genotype(self, genotype, phenotype):
        print(genotype)
        values = {}
        model = json.loads(phenotype)
        layers = model['config']['layers']
        for layer in layers:
            #print(layer)
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
            rule_index = self.parser.NT.index(f'<{key}>')

            grm_options = self.parser.GRAMMAR[f'<{key}>']
            gen_indexes = genotype[rule_index]
            fen_indexes = [grm_options.index([val]) for val in values[key]]
            print(key, values[key])
            print(gen_indexes)
            print(fen_indexes)

            genotype[rule_index] = fen_indexes[:len(gen_indexes)]

        print(genotype)
        return genotype

    def _build_right_side(self, mapping):

        blocks = None
        for block in reversed(mapping):
            name, params = block[0], block[1:]
            if name == 'maxpool':
                if blocks != None:
                    mapping.append(['upsamp', 2])
                    mapping.append(['conv', 0, 2, 1, 'same', 'relu'])
                    if ['bridge'] in blocks:
                        mapping.append(['concat', 3])
                        blocks.remove(['bridge'])
                    mapping.extend(blocks)
                blocks = []
            elif blocks != None:
                blocks.append(block)
        if blocks != None:
            if blocks != None:
                mapping.append(['upsamp', 2])
                mapping.append(['conv', 0, 2, 1, 'same', 'relu'])
                if ['bridge'] in blocks:
                    mapping.append(['concat', 3])
                    blocks.remove(['bridge'])
                mapping.extend(blocks)
        
        mapping.insert(0, ['input', (None,)+self.input_shape]) #input layer
        mapping.append(['conv', 2, 3, 1, 'same', 'relu']) #classification layer
        mapping.append(['conv', 1, 1, 1, 'same', 'sigmoid']) #output layer

        return mapping
                
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
                depth += 1
                temp = calculate_output_size(output_shape, *params[:3])
                output_shape = temp + (output_shape[2],)
            elif name == 'upsamp':
                depth -= 1
                factor = params[0]
                output_shape = (output_shape[0] * factor, output_shape[1] * factor, output_shape[2])
            elif name == 'concat':
                output_shape = (output_shape[0], output_shape[1], output_shape[2]*2)
            # print('\t'*depth, i, output_shape, block)
            outputs.append(output_shape)
        return outputs

    def _non_recursive_repair(self, mapping):
        outputs = self._get_layer_outputs(mapping)
        stack = []
        for i, layer in enumerate(mapping):
            name, params = layer[0], layer[1:]
            if name == 'maxpool':
                stack.append(outputs[i-1])
            elif name == 'upsamp' and stack != []:
                aux_output = stack.pop()
                if aux_output[:-1] == (1, 1):
                    mapping[i][1] = 1
                    #print(i, 'changing upsamp to 1x')
                #print(i, 'adjusting number of filters in layer', aux_output)
                mapping[i+1][1] = aux_output[2]

    def map_genotype_to_phenotype(self, genotype):
        
        self.naming = {}

        mapping, genotype = self.parser.dsge_recursive_parse(genotype)
        mapping = self._reshape_mapping(mapping)
        mapping = self._build_right_side(mapping)
        self._non_recursive_repair(mapping)

        model = {'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        for i, layer in enumerate(mapping):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
            model['config']['layers'].append(block)

        self._wrap_up_model(model)

        return json.dumps(model)

    def _predict_model(self, model):
        predictions = model.predict_generator(
            self.test_generator, 
            steps=self.dataset['test_steps'], 
            workers=self.workers, 
            use_multiprocessing=self.multiprocessing, 
            verbose=self.verbose)

        for i, img in enumerate(predictions):
            write_image(os.path.join(self.dataset['path'], f'test/pred/{i}.png'), img)

    def evaluate_generator(self, phenotype, predict=False):
        try:
            model = model_from_json(phenotype)

            model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)

            model = self._train_model(model)

            loss, acc = self._evaluate_model(model)

            if predict:
                self._predict_model(model)

            return loss, acc
        except Exception as e:
            print('[evaluation]', e)
            return -1, None

    def evaluate(self, solution):

        super().evaluate(solution)
        # try:
        #     if model is None:
        #         model = model_from_json(phenotype)

        #     model.compile(optimizer=self.opt, loss=self.loss, metrics=self.metrics)
        #     model.summary()

        #     x_train = self.x_train[:self.train_size]
        #     y_train = self.y_train[:self.train_size]
        #     x_valid = self.x_valid[:self.valid_size]
        #     y_valid = self.y_valid[:self.valid_size]
        #     x_test = self.x_test[:self.test_size]
        #     y_test = self.y_test[:self.test_size]

        #     ts = TimedStopping(seconds=self.timelimit, verbose=self.verbose)
        #     callbacks = [ts]

        #     if save_model:
        #         mc = ModelCheckpoint(filepath=f'{ckpt.ckpt_folder}_weights.hdf5', monitor='val_accuracy', save_weights_only=True, save_best_only=True)
        #         callbacks.append(mc)

        #     if self.training:
        #         history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, callbacks=callbacks)
        #         self.plot_loss(history)
        #         self.plot_acc(history)
        #     scores = model.evaluate(x_test, y_test, batch_size=self.batch_size, verbose=self.verbose)

        #     if self.verbose:
        #         print('scores', scores)

        #     if predict:
        #         predictions = model.predict(x_test, batch_size=self.batch_size, verbose=self.verbose)

        #         if not os.path.exists(ckpt.ckpt_folder):
        #             os.mkdir(ckpt.ckpt_folder)
                
        #         for i, img in enumerate(predictions):
        #             #img = img.astype('uint8')
        #             write_image(os.path.join(ckpt.ckpt_folder, f'{i}.png'), img)

        #     return scores, model.count_params()
        # except Exception as e:
        #     print('[evaluation]', e)
        #     return (-1, None), 0

    def plot_loss(self, history):

        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['Training Loss', 'Validation Loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.savefig(os.path.join(ckpt.ckpt_folder, 'loss.png'))

    def plot_acc(self, history):

        metric_name = self.metrics[0].__name__

        plt.figure(figsize=[8,6])
        plt.plot(history.history[metric_name])
        plt.plot(history.history['val_'+metric_name])
        plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.savefig(os.path.join(ckpt.ckpt_folder, 'acc.png'))
