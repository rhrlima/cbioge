import json
from typing import Union

from keras.models import Model, model_from_json

from ...datasets import Dataset
from ...grammars import Grammar
from ...problems import DNNProblem


class UNetProblem(DNNProblem):
    ''' Problem class for problems related to classification tasks for DNNs.
        This class includes methods focused on the design of U-Nets.'''

    def __init__(self, parser: Grammar, dataset: Dataset,
        batch_size: int=10,
        epochs: int=1,
        opt: str='adam',
        loss: Union[str, callable]='binary_crossentropy',
        metrics: list=['accuracy'],
        test_eval: bool=False,
        verbose: bool=False,
        train_args: dict=None,
        test_args: dict=None
    ):

        super().__init__(parser, dataset, batch_size, epochs, opt, loss,
            metrics, test_eval, verbose, train_args, test_args)

    def _build_right_side(self, mapping: list):

        blocks = None
        for block in reversed(mapping):
            name, _ = block[0], block[1:]
            if name == 'maxpool':
                if blocks is not None:
                    mapping.append(['upsamp', 2])
                    mapping.append(['conv', 0, 2, 1, 'same', 'relu'])
                    if ['bridge'] in blocks:
                        mapping.append(['concat', 3])
                        blocks.remove(['bridge'])
                    mapping.extend(blocks)
                blocks = []
            elif blocks is not None:
                blocks.append(block)

        if blocks is not None:
            mapping.append(['upsamp', 2])
            mapping.append(['conv', 0, 2, 1, 'same', 'relu'])
            if ['bridge'] in blocks:
                mapping.append(['concat', 3])
                blocks.remove(['bridge'])
            mapping.extend(blocks)

        return mapping

    def _calculate_output_size(self, img_shape: tuple, k: int, s: int, p: int) -> tuple:
        '''(width, height), kernel, stride, padding'''
        index = 1 if len(img_shape) == 4 else 0
        w = img_shape[index]
        h = img_shape[index+1]

        p = 0 if p == 'valid' else (k-1) / 2
        ow = ((w - k + 2 * p) // s) + 1
        oh = ((h - k + 2 * p) // s) + 1
        return (int(ow), int(oh))

    def _get_layer_outputs(self, mapping: list):
        outputs = []
        depth = 0
        for _, block in enumerate(mapping):
            name, params = block[0], block[1:]
            if name == 'input':
                output_shape = self.dataset.input_shape
            elif name == 'conv':
                output_shape = self._calculate_output_size(output_shape, *params[1:4])
                output_shape += (params[0],)
            elif name in ['maxpool', 'avgpool']:
                depth += 1
                temp = self._calculate_output_size(output_shape, *params[:3])
                output_shape = temp + (output_shape[2],)
            elif name == 'upsamp':
                depth -= 1
                factor = params[0]
                output_shape = (output_shape[0] * factor, output_shape[1] * factor, output_shape[2])
            elif name == 'concat':
                output_shape = (output_shape[0], output_shape[1], output_shape[2]*2)
            outputs.append(output_shape)
        return outputs

    def _repair(self, mapping: list):
        # changes the kernel size of pooling layers to keep image dimensions
        # as valid values (avoid reducing the size to less than 1x1)
        outputs = self._get_layer_outputs(mapping)
        stack = []
        for i, layer in enumerate(mapping):
            name, _ = layer[0], layer[1:]
            if name == 'maxpool':
                stack.append(outputs[i-1])
            elif name == 'upsamp' and stack:
                aux_output = stack.pop()
                if aux_output[:-1] == (1, 1):
                    mapping[i][1] = 1
                mapping[i+1][1] = aux_output[2]

    def _build_block(self, block_name: str, params: list, naming: dict):

        if naming is None:
            naming = dict()

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        if block_name in naming:
            naming[block_name] += 1
        else:
            naming[block_name] = 0
        name = f'{block_name}_{naming[block_name]}'

        base_block['class_name'] = self.parser.blocks[block_name][0]
        base_block['name'] = name
        for name, value in zip(self.parser.blocks[block_name][1:], params):
            base_block['config'][name] = value
        return base_block

    def _build_json_model(self, mapping: list) -> dict:

        names = dict()

        model = {'class_name': 'Model',
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        for i, layer in enumerate(mapping):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params, names)
            model['config']['layers'].append(block)

        # creates a stack with the layers that will have a bridge (concat) connection
        stack = []
        for i, layer in enumerate(model['config']['layers']):
            if layer['class_name'] in ['bridge']: #CHECK
                stack.append(model['config']['layers'][i-1]) #layer before (conv)
                model['config']['layers'].remove(model['config']['layers'][i])

        # iterates over layers and add previous layer as input of current one
        for i, layer in enumerate(model['config']['layers'][1:]):
            last = model['config']['layers'][i]
            layer['inbound_nodes'].append([[last['name'], 0, 0]])

        # creates and adds input and output layers to model
        input_layer = model['config']['layers'][0]['name']
        output_layer = model['config']['layers'][-1]['name']
        model['config']['input_layers'].append([input_layer, 0, 0])
        model['config']['output_layers'].append([output_layer, 0, 0])

        # iterates over the layers looking for the concatenate ones to then
        # adds the connection that comes from the bridge (stored in the stack)
        for i, layer in enumerate(model['config']['layers'][1:]):
            if layer['class_name'] == 'Concatenate':
                other = stack.pop()
                layer['inbound_nodes'][0].insert(0, [other['name'], 0, 0])

        return model

    def _build_model(self, mapping: list) -> Model:

        reshaped_mapping = self._reshape_mapping(mapping)

        # build right part of the network based on the left
        reshaped_mapping = self._build_right_side(reshaped_mapping)

        # insert base layers
        reshaped_mapping.insert(0, ['input', (None,)+self.dataset.input_shape]) # input layer
        reshaped_mapping.append(['conv', 2, 3, 1, 'same', 'relu']) # classification layer
        reshaped_mapping.append(['conv', 1, 1, 1, 'same', 'sigmoid']) # output layer

        # repair possible invalid connections
        self._repair(reshaped_mapping)

        # build the json structure of the model
        model = self._build_json_model(reshaped_mapping)

        # creates the model from json
        return model_from_json(json.dumps(model))
