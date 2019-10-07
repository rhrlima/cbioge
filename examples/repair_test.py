import itertools

import numpy as np

from keras.layers import *

from grammars import BNFGrammar
from problems import *


def calculate_output_size(img_shape, k, s, p):
    ''' width, height, kernel, padding, stride'''
    p = 0 if p == 'valid' else (k-1)/2
    ow = ((img_shape[0] - k + 2 * p) // s) + 1
    oh = ((img_shape[1] - k + 2 * p) // s) + 1
    return ow, oh, img_shape[2]


def is_valid_conv(input_shape, k, p, s):
    output_shape = calculate_output_size(input_shape, k, p, s)
    return (0, 0) < output_shape, output_shape


def is_valid_merge(shape_1, shape_2):
    
    return shape_1 == shape_2

 
def get_conv_combinations(parser):
    kernels = [int(i[0]) for i in parser.GRAMMAR['<ksize>']]
    strides = [int(i[0]) for i in parser.GRAMMAR['<strides>']]
    padding = [i[0] for i in parser.GRAMMAR['<padding>']]
    return list(itertools.product(kernels, strides, padding))


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


def repair(parser, genotype, mapping, input_shape, index=0, depth=0):
    '''validar casos:
    verificar shape do input com output para convoluções
    verificar compatibilidade entre junção de layers'''

    print(f"### DEPTH {depth} ###")

    if index >= len(mapping):
        # it only returns TRUE when reaches the end without errors
        return True

    #print(input_shape)
    block = mapping[index][0]
    params = mapping[index][1:]

    if block == 'conv': #consumes 6 spots
        this_config = tuple(params[1:4])

        is_valid, output_shape = is_valid_conv(input_shape, *this_config)
        if not is_valid:
            # valid, call next node
            #print(this_config, 'VALID')
            combinations = get_conv_combinations(parser)
            combinations.remove(this_config)
            #print('original', this_config)
            for comb in combinations:
                # tests a different config for the current block
                # if return is TRUE, it means it reached the end, just break loop and exit
                # it return is FALSE, try next config
                # if there's no valid config, returns FALSE (it will go up one recursive call)
                #print('testing combination:', comb)
                is_valid, output_shape = is_valid_conv(input_shape, *this_config)
                #is_valid = repair(parser, genotype, mapping, input_shape, index, depth)
                if is_valid: break
                    # if valid, no need to test other combination
                    # if false, should test next, if no next, then return false
                #print(comb, 'not valid')
            # returns false only when theres no valid combination in list
            #print('no valid combination found')
        if is_valid:
            return repair(parser, genotype, mapping, output_shape, index+1, depth+1)
        return False

    else:
        #print('no next SKIPPING')
        return repair(parser, genotype, mapping, input_shape, index+1)

    #print('ended')
    #return result
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
    #fen = problem.map_v2(gen)
    #fen = parser.dsge_recursive_parse(gen)
    fen = [['conv', 32, 3, 1, 'valid', 'linear'], ['conv', 64, 3, 1, 'valid', 'linear'], ['conv', 128, 3, 1, 'valid', 'linear'],['drop', 0.2],['dense', 10]]
    #print(fen)
    #fen = reshape_mapping(fen)
    print(fen)

    
    repaired = repair(parser, gen, fen, (4, 4, 1))
    print(repaired)