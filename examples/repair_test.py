import itertools

import numpy as np

from keras.layers import *

from grammars import BNFGrammar
from problems import *


def calculate_output_size(img_shape, k, p, s):
    ''' width, height, kernel, padding, stride'''
    p = 0 if p == 'valid' else (k-1)/2
    ow = ((img_shape[0] - k + 2 * p) / s) + 1
    oh = ((img_shape[1] - k + 2 * p) / s) + 1
    return ow, oh

def is_valid_conv(input_shape, k, p, s):
    
    return (0, 0) < calculate_output_size(input_shape, k, p, s)


def is_valid_merge(shape_1, shape_2):
    
    return shape_1 == shape_2

 
def built_block(block, params):
    base = {'class_name': None, 'config': None}

    layers = {
        'conv': ['Conv2D', 'filters', 'kernel_size', 'strides', 'padding', 'activation'],
        'avgpool': {'kernel_size': params[0], 'padding': params[1]},
        'dense': {'units': params[0]},
    }

    base['class_name'] = layers[block].pop(0)
    for name, value in zip(params, layers[block]):
        base['config'][name] = value

    return baseblocks[block]



def get_conv_combinations(parser):
    kernels = [int(i[0]) for i in parser.GRAMMAR['<ksize>']]
    padding = [i[0] for i in parser.GRAMMAR['<padding>']]
    strides = [int(i[0]) for i in parser.GRAMMAR['<strides>']]
    combinations = itertools.product(kernels, padding, strides)
    for comb in combinations:
        print(comb)
    return combinations

def reshape_mapping(mapping):

    new_mapping = []

    index = 0
    while index < len(mapping):
        if mapping[index] == 'conv':
            end = index+6
        elif mapping[index] == 'avgpool':
            end = index+3
        else:
            end = 2

        new_mapping.append(mapping[index:end])
        mapping = mapping[end:]

    return new_mapping


def repair(parser, genotype, mapping, input_shape, index=0):
    '''validar casos:
    verificar shape do input com output para convoluções
    verificar compatibilidade entre junção de layers'''

    # reached the end without errors
    if index >= len(mapping):
        return True

    block = mapping[index][0]
    params = mapping[index][1:]

    if block == 'conv': #consumes 6 spots
        #filters = int(params[0])
        kernels = int(params[1])
        strides = int(params[2])
        padding = params[3]
        #activation = params[4]

        if is_valid_conv(input_shape, kernels, padding, strides):
            # valid, call next node
            print(kernels, padding, strides, 'VALID')
            return repair(parser, genotype, mapping, input_shape, index+1)
        else:
            combinations = get_conv_combinations(parser)
            print(len(combinations))
            combinations.remove(set(kernels, padding, strides))
            print(len(combinations))
            for comb in combinations:
                is_valid = repair(parser, genotype, mapping, input_shape, index)
                if is_valid:
                    # if valid, no need to test other combination
                    # if false, should test next, if no next, then return false
                    return is_valid
            # returns false only when theres no valid combination in list
            return False

    print('SKIPPING')
    return repair(parser, genotype, mapping, input_shape, index+1)
    # elif mapping[index] == 'avgpool': #consumes 3 spots
    #     kernel = int(mapping[index+1])
    #     padding = mapping[index+2]
    #     print('AvgPool', kernel, padding)
    #     index = index+3

    # elif mapping[index] == 'dense': #consumes 2 spots
    #     units = int(mapping[index+1])
    #     print('Dense layer', units)
    #     index = index+2

    #repair(genotype, mapping, input_shape, index, depth+1)

if __name__ == '__main__':

    np.random.seed(0)

    parser = BNFGrammar('grammars/cnn2.bnf')
    problem = CNNProblem(parser)

    #gen = parser.dsge_create_solution()
    gen = [[0], [1], [2], [1], [], [0], [0], [], [0], [1], [3], [1, 0], [0], [0], [1], [2], [0], []]
    print(gen)
    fen = problem.map_v2(gen)
    #fen = parser.dsge_recursive_parse(gen)
    #print(fen)
    #fen = reshape_mapping(fen)
    #print(fen)

    #repaired = repair(parser, gen, fen, (256, 256, 1))
    #print(repaired)

    # gen = [[0], [1], [2], [1], [], [0], [0], [], [0], [1], [3], [1, 0], [0], [0], [1], [2], [0], []]
    # print(parser.dsge_recursive_parse(gen))
    # gen = [[0], [1], [2], [1], [], [0], [0], [], [0], [1], [3], [1, 0], [0], [0], [1], [2], [1], []]
    # print(parser.dsge_recursive_parse(gen))
    # gen = [[0], [1], [2], [1], [], [0], [0], [], [0], [1], [3], [1, 0], [0], [0], [1], [2], [2], []]
    # print(parser.dsge_recursive_parse(gen))

    # input_shape = (256, 256, 1)
    # repair(gen, fen, (256, 256))
    
    # print(calculate_output_size(input_shape, 1, 'valid', 1))
    # print(calculate_output_size(input_shape, 3, 'valid', 1))
    # print(calculate_output_size(input_shape, 5, 'valid', 1))
    # print(calculate_output_size(input_shape, 7, 'valid', 1))

    # print(calculate_output_size(input_shape, 5, 'same', 1))

    # print(calculate_output_size(input_shape, 5, 'valid', 2))
    # print(calculate_output_size(input_shape, 5, 'same', 2))

    # inputs = Input(input_shape)
    # print(Conv2D(filters=32, kernel_size=1, strides=1, padding='valid')(inputs))
    # print(Conv2D(filters=32, kernel_size=3, strides=1, padding='valid')(inputs))
    # print(Conv2D(filters=32, kernel_size=5, strides=1, padding='valid')(inputs))
    # print(Conv2D(filters=32, kernel_size=7, strides=1, padding='valid')(inputs))

    # print(Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(inputs))

    # print(Conv2D(filters=32, kernel_size=5, strides=2, padding='valid')(inputs))
    # print(Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(inputs))

    # print(calculate_output_size((4, 4), 1, 'valid', 1))
    # print(calculate_output_size((2, 2), 3, 'valid', 1))
    # print(calculate_output_size((1, 1), 3, 'valid', 1))

    # b = built_block('conv', [])
    # print(b)