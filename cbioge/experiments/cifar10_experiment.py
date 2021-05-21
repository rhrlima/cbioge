'''
DSGE para CIFAR 10

epochs 20
batch 32

pop 100
evals 1000
selection 5
crossover 0.8
mutation 0.01
elitism 0.1

* cross/mut halfandhalf

'''

import glob
import argparse
import json

import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from algorithms.dsge import GrammaticalEvolution
from algorithms.selection import *
from algorithms.crossover import *
from algorithms.mutation import *
from algorithms.operators import *

from grammars import BNFGrammar
from problems import CNNProblem

from utils.model import *
from utils import checkpoint as ckpt
from utils.experiments import check_os, args_evolution_exp
from utils.dataset import split_dataset


def run_evolution():

    # check if Windows to limit GPU memory and avoid errors
    check_os()

    args = args_evolution_exp()
    print(args)

    np.random.seed(args.seed)

    if args.grammar is None:
        args.grammar = 'grammars/cnn3.bnf'
    if args.dataset is None:
        args.dataset = 'datasets/cifar10.pickle'

    parser = BNFGrammar(args.grammar)
    problem = CNNProblem(parser)
    problem.read_dataset_from_pickle(args.dataset)

    problem.timelimit = args.timelimit
    problem.epochs = args.epochs
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    if not args.train is None:
        problem.train_size = args.train
    if not args.valid is None:
        problem.valid_size = args.valid
    if not args.test is None:
        problem.test_size = args.test

    selection = TournamentSelection(t_size=3, maximize=True)
    crossover = DSGECrossover(cross_rate=1.0)
    mutation = DSGENonterminalMutation(mut_rate=1.0, parser=parser, end_index=4)
    operator = HalfAndHalfOperator(op1=crossover, op2=mutation, rate=0.6)
    replace = ElitistReplacement(rate=0.25, maximize=True)

    algorithm = GrammaticalEvolution(problem, parser)

    algorithm.training = args.training
    algorithm.pop_size = args.pop
    algorithm.max_evals = args.evals

    algorithm.selection = selection
    algorithm.crossover = operator
    algorithm.replacement = replace

    ckpt.ckpt_folder = args.folder

    algorithm.verbose = (args.verbose>0) # verbose 1 or higher
    problem.verbose = (args.verbose>1) # verbose 2 or higher
    parser.verbose = (args.verbose>1)

    population = algorithm.execute(args.checkpoint)

    # remove and add better post-run
    for s in population:
        print(s.fitness, s)

if __name__ == '__main__':
    run_evolution()
