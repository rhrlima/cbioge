import argparse
import sys
import numpy as np

from keras.models import *
from keras.layers import *

from algorithms.solutions import GESolution
from algorithms import DSGECrossover, DSGEMutation

from problems import UNetProblem
from grammars import BNFGrammar


def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    args.add_argument('-n', '--number', type=int, default=0) #quantity
    args.add_argument('-v', '--verbose', type=int, default=0) #verbose

    return args.parse_args()


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
    args = get_args()

    dset = {
        'input_shape': (256, 256, 1)
    }

    parser = BNFGrammar('grammars/unet_mirror.bnf')
    problem = UNetProblem(parser, dset, None, None)

    print('creating population')
    pop = []
    for i in range(args.number):
        gen = parser.dsge_create_solution()
        fen = problem.map_genotype_to_phenotype(gen)
        solution = GESolution(gen)
        solution.phenotype = fen
        pop.append(solution)

    # print('applying crossover')
    # cross = DSGECrossover(cross_rate=0.9)
    # for i in range(10):
    #     s1 = np.random.choice(pop)
    #     s2 = np.random.choice(pop)
    #     off = cross.execute([s1, s2])
    #     for s in off:
    #         s.phenotype = problem.map_genotype_to_phenotype(s.genotype)
    #         pop.append(s)

    # print('applying mutation')
    # mut = DSGEMutation(mut_rate=0.1, parser=parser)
    # for i in range(10):
    #     s = np.random.choice(pop)
    #     mut.execute(s)
    #     s.phenotype = problem.map_genotype_to_phenotype(s.genotype)
    #     pop.append(s)

    print(len(pop))

    failed = 0
    for i, s in enumerate(pop):
        failed += not run(s, args.verbose)
        print(f'\r\r{failed}/{i+1} {failed/(i+1)}%', end='')
    print()