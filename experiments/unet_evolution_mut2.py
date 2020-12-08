import glob
import argparse
import json

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from algorithms.dsge import GrammaticalEvolution
from algorithms.selection import *
from algorithms.crossover import *
from algorithms.mutation import *
from algorithms.operators import *

from grammars import BNFGrammar
from problems import UNetProblem

from utils.model import *
from utils import checkpoint as ckpt


def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    # problem args
    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-t', '--training', type=int, default=1) #training
    args.add_argument('-trs', '--train', type=int, default=None) #train steps
    args.add_argument('-va', '--valid', type=int, default=None) #valid size
    args.add_argument('-tes', '--test', type=int, default=None) #test steos

    args.add_argument('-tl', '--timelimit', type=int, default=3600) #timelimit (in seconds) 1h
    args.add_argument('-e', '--epochs', type=int, default=1) #epochs
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-s', '--shuffle', type=int, default=0) #shuffle

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    # evolution args
    args.add_argument('-ps', '--pop', type=int, default=10) #pop
    args.add_argument('-ev', '--evals', type=int, default=20) #evals
    args.add_argument('-cr', '--crossrate', type=float, default=0.8) #crossover rate
    args.add_argument('-mr', '--mutrate', type=float, default=0.01) #mutation rate


    args.add_argument('-f', '--folder', type=str, default='checkpoints')
    args.add_argument('-c', '--checkpoint', type=int, default=0) #from checkpoint

    args.add_argument('-v', '--verbose', type=int, default=1) #verbose (1 - evolution, 2 - problem)

    args.add_argument('-rs', '--seed', type=int, default=None)

    return args.parse_args()


def run_evolution():

    import platform
    if platform.system() == 'Windows':
        limit_gpu_memory()

    args = get_args()
    print(args)

    np.random.seed(args.seed)

    parser = BNFGrammar('grammars/unet_mirror2.bnf')
    
    problem = UNetProblem(parser)

    problem.read_dataset_from_pickle(args.dataset)

    problem.timelimit = args.timelimit
    problem.epochs = args.epochs
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    wmetric = WeightedMetric(w_spe=.1, w_dic=.4, w_sen=.4, w_jac=.1)
    # problem.loss = wmetric.get_loss()
    # problem.metrics = ['accuracy', jaccard_distance, dice_coef, specificity, sensitivity]
    problem.metrics = [wmetric.get_metric()]

    if not args.train is None:
        problem.train_size = args.train
    if not args.valid is None:
        problem.valid_size = args.valid
    if not args.test is None:
        problem.test_size = args.test

    selection = TournamentSelection(t_size=5, maximize=True)
    crossover = DSGECrossover(cross_rate=args.crossrate)
    mutation = DSGENonTerminalMutation(mut_rate=args.mutrate, parser=parser, end_index=4)
    replace = ElitistReplacement(rate=0.25, maximize=True)

    algorithm = GrammaticalEvolution(problem, parser)

    algorithm.training = args.training
    algorithm.pop_size = args.pop
    algorithm.max_evals = args.evals
    algorithm.training = args.training

    algorithm.selection = selection
    algorithm.crossover = crossover
    algorithm.mutation = mutation
    algorithm.replacement = replace

    ckpt.ckpt_folder = args.folder

    algorithm.verbose = (args.verbose>0) # verbose 1 or higher
    problem.verbose = (args.verbose>1) # verbose 2 or higher
    parser.verbose = args.verbose>1

    population = algorithm.execute(args.checkpoint)

    for s in population:
        print(s.fitness, s)


if __name__ == '__main__':
    run_evolution()
