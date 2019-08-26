import sys; sys.path.append('..')
import os
import argparse

from algorithms import RandomSearch, GrammaticalEvolution
from algorithms import TournamentSelection, OnePointCrossover, PointMutation
from algorithms import GEPrune, GEDuplication, GrammaticalEvolution

from grammars import BNFGrammar

from problems import StringMatchProblem
#from utils import checkpoint


# disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_arg_parsersed():

    parser = argparse.ArgumentParser(prog='script.py')

    # not optional
    parser.add_argument('grammar', type=str)

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


if __name__ == '__main__':

    # parses the arguments
    args = get_arg_parsersed()

    # read grammar and setup parser
    parser = BNFGrammar(args.grammar)

    # problem parameters
    # problem = StringMatchProblem(parser)

    # from algorithms.solutions import GESolution
    # solution = GESolution([])
    # solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)
    # diff = problem.evaluate(solution)
    # print(diff)

    # changing pge default parameters
    # algorithm = RandomSearch(problem)

    # selection = TournamentSelection()
    # crossover = OnePointCrossover(cross_rate=0.75)
    # mutation = PointMutation(mut_rate=0.1, min_value=0, max_value=255)
    # prune = GEPrune(prun_rate=0.1)
    # duplication = GEDuplication(dupl_rate=0.1)

    # algorithm = GrammaticalEvolution(problem)
    # algorithm.maximize = True
    # algorithm.pop_size = 100
    # algorithm.max_evals = 50000
    # algorithm.max_processes = 1
    # algorithm.selection = selection
    # algorithm.crossover = crossover
    # algorithm.mutation = mutation
    # algorithm.prune = prune
    # algorithm.duplication = duplication
    # # algorithm.verbose = args.verbose

    # print('--config--')
    # # print('DATASET', args.dataset)
    # print('GRAMMAR', args.grammar)
    # # print('CKPT', args.folder, args.checkpoint)

    # print('--running--')
    # best = algorithm.execute()

    # print('--best solution--')
    # if best:
    #     print(best.fitness, best)
    #     print(best.phenotype)
    # else:
    #     print('None solution')
