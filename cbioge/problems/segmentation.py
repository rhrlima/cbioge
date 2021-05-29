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
    ''' Problem class for problems related to classification tasks for DNNs.
        This class includes methods focused on the design of U-Nets.
    '''

    def __init__(self, parser, dataset):
        super().__init__(parser, dataset)

        # segmentation specific
        self.loss = 'binary_crossentropy'

    def _wrap_up_model(self, model):
        # creates a stack with the layers that will have a bridge (concat) connection
        stack = []
        for i, layer in enumerate(model['config']['layers']):
            if layer['class_name'] in ['bridge']: #CHECK
                stack.append(model['config']['layers'][i-1]) #layer before (conv)
                model['config']['layers'].remove(model['config']['layers'][i])

        super()._wrap_up_model(model)

        # iterates over the layers looking for the concatenate ones to then 
        # adds the connection that comes from the bridge (stored in the stack)
        for i, layer in enumerate(model['config']['layers'][1:]):
            if layer['class_name'] == 'Concatenate':
                other = stack.pop()
                layer['inbound_nodes'][0].insert(0, [other['name'], 0, 0])

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

    def map_genotype_to_phenotype(self, genotype):

        mapping, genotype = self.parser.dsge_recursive_parse(genotype)
        mapping = self._reshape_mapping(mapping)

        # unet specific
        mapping = self._build_right_side(mapping)

        mapping.insert(0, ['input', (None,)+self.input_shape]) # input layer
        mapping.append(['conv', 2, 3, 1, 'same', 'relu']) # classification layer
        mapping.append(['conv', 1, 1, 1, 'same', 'sigmoid']) # output layer

        self._non_recursive_repair(mapping)

        model = self._base_build(mapping)

        self._wrap_up_model(model)

        return json.dumps(model)
