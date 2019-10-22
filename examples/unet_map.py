import sys
import numpy as np

from keras.models import *
from keras.layers import *

from problems import UNetProblem
from grammars import BNFGrammar


def run(gen, verbose=False):
    mapp = parser.dsge_recursive_parse(gen)
    rmap = problem._reshape_mapping(mapp)
    if verbose: print(gen)
    if verbose: print(rmap)
    #model = problem._map_genotype_to_phenotype(gen)
    model = problem._mirror_build(gen)
    try:
        model = model_from_json(model)
        if verbose: model.summary()
        return True
    except Exception as e:
        print(e)
        exit(1)
        return False


if __name__ == '__main__':

    #np.random.seed(42)

    dset = {
        'input_shape': (256, 256, 1)
    }

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    problem = UNetProblem(parser, dset)

    # gen = [[0], [0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1], [2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0], [0], [0, 0, 0, 0], [0, 0, 0, 0], [], [0, 0, 0, 0], [], [], [], [], []]
    # run(gen, True)

    num = int(sys.argv[1]) if len(sys.argv) == 2 else 1
    failed = 0
    for i in range(num):
        gen = parser.dsge_create_solution()
        failed += not run(gen, True)
        print(f'\r\r{failed}/{i+1} {failed/(i+1)}%', end='')
    print()
