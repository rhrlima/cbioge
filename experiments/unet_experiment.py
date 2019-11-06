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

    args.add_argument('-trs', type=int, default=5) #train steps
    args.add_argument('-tes', type=int, default=5) #test steos
    args.add_argument('-aug', type=int, default=0) #augmentation
    args.add_argument('-b', type=int, default=5) #batch
    args.add_argument('-s', type=int, default=0) #shuffle
    args.add_argument('-v', type=int, default=1) #verbose

    return args.parse_args()


def run():
    np.random.seed(0)

    args = get_args()

    print(args)

    dset_args = {
        "path": "datasets/membrane",
        "train_path": "datasets/membrane/train_posproc",
        "test_path": "datasets/membrane/test_posproc",
        "input_shape": (256, 256, 1),
        "train_steps": args.trs,
        "test_steps": args.tes,
        "aug": dict(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')
    }

    data_aug = ImageDataGenerator(**dset_args['aug']) if args.aug > 0 else None
    train_gen = DataGenerator(dset_args['train_path'], dset_args['input_shape'], batch_size=args.b, data_aug=data_aug, shuffle=args.s)
    test_gen = DataGenerator(dset_args['test_path'], dset_args['input_shape'], batch_size=args.b, shuffle=args.s)

    problem = UNetProblem(None, dset_args)
    problem.train_generator = train_gen
    problem.test_generator = test_gen
    problem.verbose = args.v

    solution = GESolution([])
    solution.phenotype = unet(dset_args['input_shape']).to_json()

    result = problem.evaluate(solution.phenotype)
    print(result)


if __name__ == '__main__':
    run()