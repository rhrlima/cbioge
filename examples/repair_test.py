import numpy as np

from keras.layers import *

from grammars import BNFGrammar
from problems import CnnProblem
from problems import ImageSegmentationProblem

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
    blocks = {
        'conv': {},
        'avgpool': ,
        'dense': ,
    }

    return blocks[block](32, 3)


def repair(genotype, mapping, input_shape, index=0, depth=0):
    '''validar casos:
    verificar shape do input com output para convoluções
    verificar compatibilidade entre junção de layers'''
    if index == len(mapping):
        return

    if mapping[index] == 'conv': #consumes 6 spots
        filters = int(mapping[index+1])
        kernel = int(mapping[index+2])
        stride = int(mapping[index+3])
        padding = mapping[index+4]
        activation = mapping[index+5]
        print('Conv layer', filters, kernel, stride, padding, activation)
        print('valid', is_valid_conv(input_shape, kernel, padding, stride))
        index = index+6

    elif mapping[index] == 'avgpool': #consumes 3 spots
        kernel = int(mapping[index+1])
        padding = mapping[index+2]
        print('AvgPool', kernel, padding)
        index = index+3

    elif mapping[index] == 'dense': #consumes 2 spots
        units = int(mapping[index+1])
        print('Dense layer', units)
        index = index+2

    repair(genotype, mapping, input_shape, index, depth+1)

    pass

if __name__ == '__main__':

    np.random.seed(0)

    parser = BNFGrammar('grammars/cnn2.bnf')
    problem = CnnProblem(parser)

    gen = parser.dsge_create_solution()
    print(gen)
    #fen = problem.map_v2(gen)
    fen = parser.dsge_recursive_parse(gen)
    print(fen)

    input_shape = (256, 256, 1)
    repair(gen, fen, (256, 256))
    
    print(calculate_output_size(input_shape, 1, 'valid', 1))
    print(calculate_output_size(input_shape, 3, 'valid', 1))
    print(calculate_output_size(input_shape, 5, 'valid', 1))
    print(calculate_output_size(input_shape, 7, 'valid', 1))

    print(calculate_output_size(input_shape, 5, 'same', 1))

    print(calculate_output_size(input_shape, 5, 'valid', 2))
    print(calculate_output_size(input_shape, 5, 'same', 2))

    inputs = Input(input_shape)
    print(Conv2D(filters=32, kernel_size=1, strides=1, padding='valid')(inputs))
    print(Conv2D(filters=32, kernel_size=3, strides=1, padding='valid')(inputs))
    print(Conv2D(filters=32, kernel_size=5, strides=1, padding='valid')(inputs))
    print(Conv2D(filters=32, kernel_size=7, strides=1, padding='valid')(inputs))

    print(Conv2D(filters=32, kernel_size=5, strides=1, padding='same')(inputs))

    print(Conv2D(filters=32, kernel_size=5, strides=2, padding='valid')(inputs))
    print(Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(inputs))

    print(calculate_output_size((4, 4), 1, 'valid', 1))
    print(calculate_output_size((2, 2), 3, 'valid', 1))
    print(calculate_output_size((1, 1), 3, 'valid', 1))

    b = built_block('conv', [])
    print(b)