import sys; sys.path.append('..')
import os
import argparse

from algorithms import RandomSearch
from grammars import BNFGrammar
from problems import CnnProblem
from utils import checkpoint


# disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_arg_parsersed():

    parser = argparse.ArgumentParser(prog='script.py')

    parser.add_argument('grammar', type=str)
    parser.add_argument('dataset', type=str)

    parser.add_argument('-f', '--folder', dest='folder', default='checkpoints')
    parser.add_argument('-c', '--checkpoint', default=True, type=str2bool)

    parser.add_argument('-sd', '--seed', default=None, type=int)

    parser.add_argument('-ep', '--epochs', default=1, type=int)
    parser.add_argument('-b', '--batch', default=128, type=int)
    parser.add_argument('-v', '--verbose', default=0, type=int)

    parser.add_argument('-e', '--evals', default=20, type=int)

    parser.add_argument('-min', '--mingenes', default=2, type=int)
    parser.add_argument('-max', '--maxgenes', default=10, type=int)

    parser.add_argument('-mp', '--maxprocesses', default=8, type=int)

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

    # read grammar and setup parser
    parser = BNFGrammar(args.grammar)

    # problem dataset and parameters
    problem = CnnProblem(parser, args.dataset)
    problem.batch_size = args.batch
    problem.epochs = args.epochs

    # checkpoint folder
    checkpoint.ckpt_folder = args.folder

    # changing pge default parameters
    algorithm = RandomSearch(problem)

    algorithm.MIN_VALUE = 0
    algorithm.MAX_VALUE = 255
    algorithm.MIN_SIZE = 2
    algorithm.MAX_SIZE = 20
    algorithm.MAX_EVALS = 10
    algorithm.MAX_PROCESSES = 2

    print('--config--')
    print('DATASET', args.dataset)
    print('GRAMMAR', args.grammar)
    print('CKPT', args.folder, args.checkpoint)

    print('--running--')
    best = algorithm.execute()

    print('--best solution--')
    if best:
        print(best.fitness, best)
    else:
        print('None solution')
