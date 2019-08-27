import sys; sys.path.append('..')
import os
import argparse

from algorithms import RandomSearch, GrammaticalEvolution
from algorithms import TournamentSelection, OnePointCrossover, PointMutation
from algorithms import GEPrune, GEDuplication, GrammaticalEvolution
from grammars import BNFGrammar

from problems import SymbolicRegressionProblem, StringMatchProblem


def get_arg_parsersed():

    parser = argparse.ArgumentParser(prog='script.py')

    # not optional
    parser.add_argument('grammar', type=str)

    # checkpoint
    parser.add_argument('-f', '--folder', dest='folder', default='checkpoints')
    parser.add_argument('-c', '--checkpoint', default=True, type=str2bool)

    # problem
    parser.add_argument('-sd', '--seed', default=None, type=int)
    parser.add_argument('-ep', '--epochs', default=1, type=int)
    parser.add_argument('-b', '--batch', default=128, type=int)
    parser.add_argument('-v', '--verbose', default=0, type=int)

    # algorithm
    parser.add_argument('-e', '--evals', default=20, type=int)
    parser.add_argument('-min', '--mingenes', default=2, type=int)
    parser.add_argument('-max', '--maxgenes', default=10, type=int)

    parser.add_argument('-mp', '--maxprocesses', default=2, type=int)

    return parser.parse_args()


def equation(x):

    return x**4 + x**3 + x**2 + x + 1.0


if __name__ == '__main__':

    parser = BNFGrammar('../grammars/reg.bnf')

    problem = SymbolicRegressionProblem(parser, equation)

    alg = GrammaticalEvolution(problem)
    alg.maximize = False
    alg.pop_size = 100
    alg.max_evals = 50000
    alg.max_processes = 1
    alg.selection = TournamentSelection()
    alg.crossover = OnePointCrossover(cross_rate=0.75)
    alg.mutation = PointMutation(mut_rate=0.1, min_value=0, max_value=255)
    alg.prune = GEPrune(prun_rate=0.1)
    alg.duplication = GEDuplication(dupl_rate=0.1)

    print('--running--')
    best = alg.execute()

    print('--best solution--')
    if best:
        print(best.fitness, best)
        print(best.phenotype)
    else:
        print('None solution')
