import sys
import numpy as np

from keras.models import *
from keras.layers import *

from problems import UNetProblem
from grammars import BNFGrammar


def run(gen, verbose=False):    
    if verbose: print(gen)
    fen = parser.dsge_recursive_parse(gen)
    fen = problem._reshape_mapping(fen)
    fen = problem._build_right_side(fen)
    out = problem._list_layer_outputs(fen)
    problem._non_recursive_repair(fen, out)
    out = problem._list_layer_outputs(fen)
    
    model = problem._map_genotype_to_phenotype(gen, fen)
    
    try:
        model = model_from_json(model)
        if verbose: model.summary()
        return True
    except Exception as e:
        print(e)
        return False


if __name__ == '__main__':

    #np.random.seed(42)

    dset = {
        'input_shape': (256, 256, 1)
    }

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    problem = UNetProblem(parser, dset)

    gen = [[0], [0, 2, 0, 2, 0, 2, 0, 2], [0, 0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [], [1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0]]
    run(gen, True)

    num = int(sys.argv[1]) if len(sys.argv) == 2 else 0
    failed = 0
    for i in range(num):
        gen = parser.dsge_create_solution()
        failed += not run(gen, True)
        print(f'\r\r{failed}/{i+1} {failed/(i+1)}%', end='')
    print()
