import json
import argparse

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from algorithms.solutions import GESolution
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

from utils.model import *

def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-tr', '--train', type=int, default=None) #train size
    args.add_argument('-va', '--valid', type=int, default=None) #valid size
    args.add_argument('-te', '--test', type=int, default=None) #test size

    args.add_argument('-p', '--predict', type=int, default=0) #predict
    args.add_argument('-b', '--batch', type=int, default=1) #batch

    args.add_argument('-v', '--verbose', type=int, default=1) #verbose

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    return args.parse_args()


def run():

    #np.random.seed(0)

    args = get_args()

    print(args)

    problem = UNetProblem(None)
    problem.read_dataset_from_pickle(args.dataset)

    if not args.train is None:
        problem.train_size = args.train
    if not args.valid is None:
        problem.valid_size = args.valid
    if not args.test is None:
        problem.test_size = args.test

    problem.verbose = args.verbose
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    solution = GESolution([])
    solution.phenotype = unet(problem.input_shape).to_json()

    for _ in range(5):
        result = problem.evaluate(solution.phenotype, args.predict)
        print(result)


if __name__ == '__main__':
    run()
