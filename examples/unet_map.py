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
    print(fen)
    for block in fen:
        if block[0] == 'maxpool':
            return False
    return True
    #model = problem.map_genotype_to_phenotype(gen)
    # try:
    #     model = model_from_json(model)
    #     if verbose: model.summary()
    #     return True
    # except Exception as e:
    #     print(e)
    #     return False


if __name__ == '__main__':

    #np.random.seed(42)

    dset = {
        'input_shape': (256, 256, 1)
    }

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    problem = UNetProblem(parser, dset)

    failed = 0

    gen = [[0], [0, 2, 0, 2, 0, 2, 0, 2], [0, 0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0], [5, 5, 6, 6, 7, 7, 8, 8, 9, 9], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [], [1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0]]
    failed +=run(gen, True)

    num = int(sys.argv[1]) if len(sys.argv) == 2 else 0

    for i in range(num):
        failed += not run(parser.dsge_create_solution())
        print(f'\r\r{failed}/{i+1} {failed/(i+1)}%', end='')
    print()
