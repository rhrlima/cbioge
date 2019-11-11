import sys
import numpy as np

from keras.models import *
from keras.layers import *

from algorithms.solutions import GESolution
from algorithms import DSGECrossover, DSGEMutation

from problems import UNetProblem
from grammars import BNFGrammar


def run(solution, verbose=False):    
    if verbose: print(solution.genotype)
    try:
        model = model_from_json(solution.phenotype)
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
    problem = UNetProblem(parser, dset, None, None)

    failed = 0

    num = int(sys.argv[1]) if len(sys.argv) == 2 else 0

    pop = []
    for i in range(num):
        gen = parser.dsge_create_solution()
        fen = problem.map_genotype_to_phenotype(gen)
        solution = GESolution(gen)
        solution.phenotype = fen

        pop.append(solution)

    #     print(f'\r\r{failed}/{i+1} {failed/(i+1)}%', end='')
    # print()

    # print('applying crossover')
    # for i in range(10):
    #     cross = DSGECrossover(cross_rate=0.9)
    #     s1 = np.random.choice(pop)
    #     s2 = np.random.choice(pop)
    #     off = cross.execute([s1, s2])

    #     pop += off

    # print('applying mutation')
    # mut = DSGEMutation(mut_rate=0.1, parser=parser)
    # for i, s in enumerate(pop):
    #     mut.execute(s)
    #     fen = problem.map_genotype_to_phenotype(s.genotype)

    #     pop.append(s)

    # print(len(pop))
