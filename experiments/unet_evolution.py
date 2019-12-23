import glob
import argparse
import json

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from algorithms.dsge import GrammaticalEvolution
from algorithms import TournamentSelection, DSGECrossover, DSGEMutation, ReplaceWorst
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

from utils.model import *


def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    #args.add_argument('name', type=str) #name
    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-t', '--training', type=int, default=1) #apply training
    args.add_argument('-trs', '--train', type=int, default=None) #train steps
    args.add_argument('-va', '--valid', type=int, default=None) #valid size
    args.add_argument('-tes', '--test', type=int, default=None) #test steos

    args.add_argument('-e', '--epochs', type=int, default=1) #epochs
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-s', '--shuffle', type=int, default=0) #shuffle

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    args.add_argument('-ps', '--pop', type=int, default=5) #pop
    args.add_argument('-ev', '--evals', type=int, default=10) #evals

    args.add_argument('-v', '--verbose', type=int, default=1) #verbose (1 - evolution, 2 - problem)

    args.add_argument('-rs', '--seed', type=int, default=None)

    return args.parse_args()


if __name__ == '__main__':

    args = get_args()
    print(args)

    np.random.seed(args.seed)

    parser = BNFGrammar('grammars/unet_mirror2.bnf')
    
    problem = UNetProblem(parser)

    problem.read_dataset_from_pickle(args.dataset)

    problem.verbose = (args.verbose>1) # verbose 2 or higher
    problem.epochs = args.epochs
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    problem.loss = weighted_measures_loss
    problem.metrics = [weighted_measures]

    if not args.train is None:
        problem.train_size = args.train
    if not args.valid is None:
        problem.valid_size = args.valid
    if not args.test is None:
        problem.test_size = args.test

    selection = TournamentSelection(t_size=2, maximize=True)
    crossover = DSGECrossover(cross_rate=0.9)
    mutation = DSGEMutation(mut_rate=0.01, parser=parser)
    replace = ReplaceWorst(maximize=True)

    algorithm = GrammaticalEvolution(problem, parser)

    algorithm.training = args.training
    algorithm.pop_size = args.pop
    algorithm.max_evals = args.evals

    algorithm.selection = selection
    algorithm.crossover = crossover
    algorithm.mutation = mutation
    algorithm.replacement = replace

    algorithm.verbose = (args.verbose>0) # verbose 1 or higher

    population = algorithm.execute()

    for s in population:
        print(s.fitness, s)
