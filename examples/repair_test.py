import itertools

import numpy as np

from keras.layers import *
from keras.models import model_from_json

from grammars import BNFGrammar
from problems import *


def calculate_output_size(img_shape, k, s, p):
    ''' width, height, kernel, padding, stride'''
    p = 0 if p == 'valid' else (k-1) // 2
    ow = ((img_shape[0] - k + 2 * p) // s) + 1
    oh = ((img_shape[1] - k + 2 * p) // s) + 1
    return ow, oh, img_shape[1:]


def is_valid_conv(input_shape, k, s, p):
    


def is_valid_merge(shape_1, shape_2):
    
    return shape_1 == shape_2

 
def get_conv_configurations(parser, max_img_size):
    kernels = [i[0] for i in parser.GRAMMAR['<ksize>']]
    strides = [i[0] for i in parser.GRAMMAR['<strides>']]
    padding = [i[0] for i in parser.GRAMMAR['<padding>']]
    conv_configs = list(itertools.product(kernels, strides, padding))
    conv_valid_configs = {}
    for img_size in range(1, max_img_size):
        key = str(img_size)
        conv_valid_configs[key] = conv_configs[:] #copies the configs list
        for config in conv_configs:
            if calculate_output_size((img_size, img_size), *config) <= (0, 0):
                conv_valid_configs[key].remove(config)
    return conv_valid_configs


def get_configurations_for_size(img_shape):
    indexes = range(len(conv_configs[str(img_shape[0])]))
    np.random.shuffle(indexes)
    return indexes


def _repair_conv(mapping, input_shape, index=0, depth=0, possibilities=None):

    print('#'*depth, depth)

    if mapping[index][0] != 'conv' or index >= len(mapping):
        return True

    this_config = tuple(mapping[index][2:5])


    is_valid = this_config in conv_configs[str()]
    #, output_shape = is_valid_conv(input_shape, *this_config)
    if is_valid:
        return _repair_conv(mapping, output_shape, index+1, depth+1)
    else:
        
        if possibilities is None:
            possibilities = get_conv_configurations()

        possibilities.remove(this_config)

        for new_config in possibilities:
            mapping[index][2:5] = list(new_config)
            is_valid = _repair_conv(mapping, input_shape, index, depth, possibilities)
            if is_valid:
                return True
        
    print('#'*depth, depth)
    return False


def repair(parser, genotype, mapping, input_shape, index=0, depth=0):
    '''validar casos:
    verificar shape do input com output para convoluções
    verificar compatibilidade entre junção de layers'''

    if index >= len(mapping):
        return True, mapping

    valid = False
    if mapping[index][0] == 'conv': #consumes 6 spots
        valid = _repair_conv(mapping, input_shape, index, depth)

    valid, mapping = repair(parser, genotype, mapping, input_shape, index+1, depth+1)
    
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
    return valid, mapping


if __name__ == '__main__':

    #np.random.seed(0)
    input_shape = (8, 8, 1)

    parser = BNFGrammar('grammars/cnn2.bnf')
    problem = CNNProblem(parser)
    problem.input_shape = (None,) + input_shape

    gen = parser.dsge_create_solution()
    fen = parser.dsge_recursive_parse(gen)
    print(gen)
    print(fen)

    global conv_configs
    conv_configs = get_conv_configurations(parser, 20)
    # for _ in range(10):
    #     gen = parser.dsge_create_solution()
    #     fen = parser.dsge_recursive_parse(gen)
    #     fen = problem._reshape_mapping(fen)
    #     #fen = problem.map_v2(gen)
    #     #print(model_from_json(fen) != None)
        
    #     copy_fen = fen
    #     repaired, fen = repair(parser, gen, fen, input_shape)
    #     if repaired:
    #         model = problem.map_v2(gen, fen)
    #         model = model_from_json(model)
    #         if not model: print('DID NOT COMPILED')
    #     else:
    #         print('NOTOK')
    #         print(gen)
    #         print(copy_fen)
    #         print(fen)
    #         exit(1)
