import json
import argparse

import numpy as np

from algorithms.solutions import GESolution
from grammars import BNFGrammar
from problems import UNetProblem

from utils.model import *

from keras.models import model_from_json

def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-s', '--solution', type=str, default='[]') #dataset

    args.add_argument('-t', '--training', type=int, default=1) #apply training
    args.add_argument('-tr', '--train', type=int, default=None) #train size
    args.add_argument('-va', '--valid', type=int, default=None) #valid size
    args.add_argument('-te', '--test', type=int, default=None) #test size

    args.add_argument('-tl', '--timelimit', type=int, default=3600) #timelimit
    args.add_argument('-e', '--epochs', type=int, default=1) #epochs
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-p', '--predict', type=int, default=0) #predict

    args.add_argument('-v', '--verbose', type=int, default=1) #verbose 

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    args.add_argument('-rs', '--seed', type=int, default=None)

    return args.parse_args()


def run():

    args = get_args()
    print(args)

    np.random.seed(args.seed)

    parser = BNFGrammar('grammars/unet_mirror2.bnf')
    problem = UNetProblem(parser)
    problem.read_dataset_from_pickle(args.dataset)

    if not args.train is None:
        problem.train_size = args.train
    if not args.valid is None:
        problem.valid_size = args.valid
    if not args.test is None:
        problem.test_size = args.test

    problem.timelimit = args.timelimit
    problem.epochs = args.epochs
    problem.verbose = args.verbose
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    problem.loss = weighted_measures_loss
    problem.metrics = [weighted_measures]

    s_values = eval(args.solution)
    if s_values == []:
        print('Empty solution. Running original UNET')
        solution = GESolution([])
        solution.phenotype = unet(problem.input_shape).to_json()
    else:
        print('Running solution', s_values)
        solution = GESolution(s_values)
        solution.phenotype = problem.map_genotype_to_phenotype(solution.genotype)

    for _ in range(10): #TEMP
        result = problem.evaluate(solution.phenotype, train=args.training, predict=args.predict)
        print(result)


if __name__ == '__main__':
    run()
