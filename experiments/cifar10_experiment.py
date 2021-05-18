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


def load_cifar10_from_keras():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    pixel_depth = 255
    x_train = x_train / pixel_depth
    x_test = x_test / pixel_depth

    x_valid, y_valid, x_train, y_train = split_dataset(x_train, y_train, 10000)
    
    print(x_train[0].min(), y_train[0].max())
    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    data_dict = {
        'x_train': x_train,
        'y_train': y_train,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'x_test': x_test,
        'y_test': y_test,
        'input_shape': x_train[0].shape,
        'num_classes': 10
    }

    return data_dict


def run_evolution():

    # check if Windows to limit GPU memory and avoid errors
    check_os()

    args = args_evolution_exp()
    print(args)

    np.random.seed(args.seed)

    if args.grammar is None:
        args.grammar = 'grammars/cnn2.bnf'

    parser = BNFGrammar(args.grammar)
    problem = CNNProblem(parser)
    problem.read_dataset_from_dict(load_cifar10_from_keras())

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

    #solution = parser.dsge_create_solution()
    #phenotype = problem.map_genotype_to_phenotype(solution)
    #problem.evaluate(phenotype=phenotype, predict=True, save_model=True)

    population = algorithm.execute(args.checkpoint)

    # remove and add better post-run
    for s in population:
        print(s.fitness, s)


if __name__ == '__main__':
    run_evolution()
