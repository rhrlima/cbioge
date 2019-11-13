import json
import argparse

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from algorithms.solutions import GESolution
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

from examples.unet_model import *

def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-tr', '--train', type=int, default=5) #train steps
    args.add_argument('-te', '--test', type=int, default=5) #test steos

    args.add_argument('-p', '--predict', type=int, default=0) #predict
    args.add_argument('-b', '--batch', type=int, default=1) #batch

    args.add_argument('-v', '--verbose', type=int, default=1) #verbose

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    return args.parse_args()


def run():
    args = get_args()

    print(args)

    problem = UNetProblem(None)
    problem.read_dataset_from_pickle(args.dataset)
    problem.verbose = args.verbose
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    solution = GESolution([])
    solution.phenotype = unet(dset_args['input_shape']).to_json()

    result = problem.evaluate2(solution.phenotype)
    print(result)


if __name__ == '__main__':
    run()
