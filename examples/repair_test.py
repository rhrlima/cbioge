import itertools

import numpy as np

from keras.layers import *
from keras.models import model_from_json

from grammars import BNFGrammar
from problems import *
from utils.image import *


if __name__ == '__main__':

    #np.random.seed(0)
    input_shape = (10, 10, 1)

    parser = BNFGrammar('grammars/cnn2.bnf')
    problem = CNNProblem(parser)
    problem.input_shape = (None,) + input_shape
    problem._generate_configurations()
    
    for _ in range(10):
        gen = parser.dsge_create_solution()
        #print(gen)
        fen = problem._map_genotype_to_phenotype(gen)
        #print(fen)
        if fen:
            model = model_from_json(fen)
            print(not model is None)
            if not model:
                print('DID NOT COMPILED')
                exit(1)
        else:
            print('NOT VALID')
            exit(1)