import sys; sys.path.append('..')
import os
import argparse

from algorithms import GrammaticalEvolution
from algorithms import TournamentSelection
from algorithms import OnePointCrossover
from algorithms import PointMutation
from algorithms import GEPrune
from algorithms import GEDuplication
from grammars import BNFGrammar
from problems import CnnProblem
from utils import checkpoint


# disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_arg_parsersed():

    parser = argparse.ArgumentParser(prog='script.py')

    # not optional
    parser.add_argument('grammar', type=str)
    parser.add_argument('dataset', type=str)

    # checkpoint
    parser.add_argument('-f', '--folder', dest='folder', default='checkpoints')
    parser.add_argument('-c', '--checkpoint', default=True, type=str2bool)

    # problem
    parser.add_argument('-sd', '--seed', default=None, type=int)
    parser.add_argument('-ep', '--epochs', default=1, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)
    parser.add_argument('-v', '--verbose', default=False, type=str2bool)

    # algorithm
    parser.add_argument('-p', '--population', default=5, type=int)
    parser.add_argument('-e', '--evals', default=10, type=int)
    parser.add_argument('-cr', '--crossover', default=0.8, type=float)
    parser.add_argument('-mt', '--mutation', default=0.1, type=float)
    parser.add_argument('-pr', '--prune', default=0.1, type=float)
    parser.add_argument('-dp', '--duplication', default=0.1, type=float)
    parser.add_argument('-mig', '--mingenes', default=2, type=int)
    parser.add_argument('-mag', '--maxgenes', default=10, type=int)
    parser.add_argument('-mp', '--maxprocesses', default=2, type=int)

    return parser.parse_args()


def str2bool(value):
    if value.lower() in ('true', 't', '1'):
        return True
    elif value.lower() in ('false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    # parses the arguments
    args = get_arg_parsersed()

    # checkpoint folder
    checkpoint.ckpt_folder = args.folder

    # read grammar and setup parser
    parser = BNFGrammar(args.grammar)

    # problem dataset and parameters
    problem = CnnProblem(parser, args.dataset)
    problem.batch_size = args.batch
    problem.epochs = args.epochs

    # genetic operators to GE
    selection = TournamentSelection(maximize=True)
    crossover = OnePointCrossover(cross_rate=args.crossover)
    mutation = PointMutation(mut_rate=args.mutation,
                             min_value=0, max_value=255)
    prune = GEPrune(prun_rate=args.prune)
    duplication = GEDuplication(dupl_rate=args.duplication)

    # changing ge default parameters
    alg = GrammaticalEvolution(problem)
    alg.seed = args.seed
    alg.pop_size = args.population
    alg.max_evals = args.evals
    alg.min_genes = args.mingenes
    alg.max_genes = args.maxgenes
    alg.max_processes = args.maxprocesses
    alg.selection = selection
    alg.crossover = crossover
    alg.mutation = mutation
    alg.prune = prune
    alg.duplication = duplication
    alg.verbose = args.verbose

    print('--config--')
    print('DATASET', args.dataset)
    print('GRAMMAR', args.grammar)
    print('EPOCHS', args.epochs)
    print('BATCH', args.batch)
    print('CKPT', args.folder, args.checkpoint)

    print('SEED', args.seed)
    print('POP', args.population)
    print('EVALS', args.evals)
    print(f'CROSS({args.crossover}) MUT({args.mutation}) ',
          f'PRUN({args.prune}) DUPL({args.duplication})')

    print('--running--')
    best = alg.execute(checkpoint=args.checkpoint)

    print('--best solution--')
    if best:
        print(best.fitness, best)
    else:
        print('None solution')
