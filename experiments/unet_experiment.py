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

    args.add_argument('-trs', '--train', type=int, default=5) #train steps
    args.add_argument('-tes', '--test', type=int, default=5) #test steos
    args.add_argument('-aug', '--augment', type=int, default=0) #augmentation
    args.add_argument('-p', '--predict', type=int, default=0) #predict
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-s', '--shuffle', type=int, default=0) #shuffle
    args.add_argument('-v', '--verbose', type=int, default=1) #verbose

    args.add_argument('-w', '--workers', type=int, default=1) #workers    
    args.add_argument('-mp', '--multip', type=int, default=0) #multiprocessing

    return args.parse_args()


def run():
    args = get_args()

    print(args)

    dset_args = json.loads(open(args.dataset, 'r').read())
    dset_args['train_steps'] = args.train
    dset_args['test_steps'] = args.test

    data_aug = ImageDataGenerator(**dset_args['aug']) if args.augment > 0 else None
    train_gen = DataGenerator(dset_args['train_path'], dset_args['input_shape'], batch_size=args.batch, data_aug=data_aug, shuffle=args.shuffle)
    test_gen = DataGenerator(dset_args['test_path'], dset_args['input_shape'], batch_size=args.batch, shuffle=args.shuffle)

    problem = UNetProblem(None, dset_args, train_gen, test_gen)
    problem.verbose = args.verbose
    problem.workers = args.workers
    problem.multiprocessing = args.multip

    solution = GESolution([])
    solution.phenotype = unet(dset_args['input_shape']).to_json()

    result = problem.evaluate(solution.phenotype, predict=args.predict)
    print(result)


if __name__ == '__main__':
    run()
