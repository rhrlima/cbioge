import json

from keras.models import Model, model_from_json
from cbioge.datasets import Dataset
from cbioge.grammars import Grammar
from cbioge.problems import DNNProblem
from cbioge.algorithms import GESolution
from cbioge.utils.image import calculate_output_size


class UNetProblem(DNNProblem):
    ''' Problem class for problems related to classification tasks for DNNs.
        This class includes methods focused on the design of U-Nets.
    '''
    def __init__(self, parser: Grammar, dataset: Dataset, 
        batch_size=10, 
        epochs=1, 
        #timelimit=None, 
        test_eval=False, 
        verbose=False, 
        **kwargs):

        super().__init__(parser, dataset, 
            batch_size, epochs, test_eval, verbose, **kwargs)

        # segmentation specific
        self.loss = 'binary_crossentropy'

    def _build_right_side(self, mapping):
        blocks = None
        for block in reversed(mapping):
            name, _ = block[0], block[1:]
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

        return mapping

    def _get_layer_outputs(self, mapping):
        outputs = []
        depth = 0
        for i, block in enumerate(mapping):
            name, params = block[0], block[1:]
            if name == 'input':
                output_shape = self.dataset.input_shape
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
        # changes the kernel size from pooling layers to keep image dimensions
        # as valid values (avoid reducing the size to less than 1x1)
        outputs = self._get_layer_outputs(mapping)
        stack = []
        for i, layer in enumerate(mapping):
            name, _ = layer[0], layer[1:]
            if name == 'maxpool':
                stack.append(outputs[i-1])
            elif name == 'upsamp' and stack != []:
                aux_output = stack.pop()
                if aux_output[:-1] == (1, 1):
                    mapping[i][1] = 1
                    #print(i, 'changing upsamp to 1x')
                #print(i, 'adjusting number of filters in layer', aux_output)
                mapping[i+1][1] = aux_output[2]

    def _build_block(self, block_name, params):

        base_block = {'class_name': None, 'name': None, 'config': {}, 'inbound_nodes': []}

        if block_name in self.naming:
            self.naming[block_name] += 1
        else:
            self.naming[block_name] = 0
        name = f'{block_name}_{self.naming[block_name]}'

        base_block['class_name'] = self.parser.blocks[block_name][0]
        base_block['name'] = name
        for name, value in zip(self.parser.blocks[block_name][1:], params):
            base_block['config'][name] = value
        return base_block

    def _build(self, mapping):

        self.naming = {}

        model = {'class_name': 'Model', 
            'config': {'layers': [], 'input_layers': [], 'output_layers': []}}

        for i, layer in enumerate(mapping):
            block_name, params = layer[0], layer[1:]
            block = self._build_block(block_name, params)
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

    # def _wrap_up_model(self, model):
    #     # iterates over layers and add previous layer as input of current one
    #     for i, layer in enumerate(model['config']['layers'][1:]):
    #         last = model['config']['layers'][i]
    #         layer['inbound_nodes'].append([[last['name'], 0, 0]])

    #     # creates and adds input and output layers to model
    #     input_layer = model['config']['layers'][0]['name']
    #     output_layer = model['config']['layers'][-1]['name']
    #     model['config']['input_layers'].append([input_layer, 0, 0])
    #     model['config']['output_layers'].append([output_layer, 0, 0])

    # def _wrap_up_model(self, model):
    #     # creates a stack with the layers that will have a bridge (concat) connection
    #     stack = []
    #     for i, layer in enumerate(model['config']['layers']):
    #         if layer['class_name'] in ['bridge']: #CHECK
    #             stack.append(model['config']['layers'][i-1]) #layer before (conv)
    #             model['config']['layers'].remove(model['config']['layers'][i])

    #     # iterates over layers and add previous layer as input of current one
    #     for i, layer in enumerate(model['config']['layers'][1:]):
    #         last = model['config']['layers'][i]
    #         layer['inbound_nodes'].append([[last['name'], 0, 0]])

    #     # creates and adds input and output layers to model
    #     input_layer = model['config']['layers'][0]['name']
    #     output_layer = model['config']['layers'][-1]['name']
    #     model['config']['input_layers'].append([input_layer, 0, 0])
    #     model['config']['output_layers'].append([output_layer, 0, 0])

    #     # iterates over the layers looking for the concatenate ones to then 
    #     # adds the connection that comes from the bridge (stored in the stack)
    #     for i, layer in enumerate(model['config']['layers'][1:]):
    #         if layer['class_name'] == 'Concatenate':
    #             other = stack.pop()
    #             layer['inbound_nodes'][0].insert(0, [other['name'], 0, 0])

    def map_genotype_to_phenotype(self, solution: GESolution) -> Model:

        mapping = self.parser.dsge_recursive_parse(solution.genotype)

        # build right part of the network based on the left
        mapping = self._build_right_side(mapping)

        # insert base layers
        mapping.insert(0, ['input', (None,)+self.dataset.input_shape]) # input layer
        mapping.append(['conv', 2, 3, 1, 'same', 'relu']) # classification layer
        mapping.append(['conv', 1, 1, 1, 'same', 'sigmoid']) # output layer

        # repair possible invalid connections
        self._non_recursive_repair(mapping)

        # build the json structure of the model
        model = self._build(mapping)

        #self._wrap_up_model(model)

        # return json.dumps(model)
        model = model_from_json(json.dumps(model))

        if model is not None:
            solution.phenotype = model.to_json()
            solution.data['params'] = model.count_params()
        else:
            solution.phenotype = None
            solution.data['params'] = 0
        solution.data['mapping'] = mapping

        return model