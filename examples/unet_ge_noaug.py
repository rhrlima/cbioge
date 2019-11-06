import os
import glob

from keras.preprocessing.image import ImageDataGenerator

from algorithms.solutions import GESolution
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

from examples.unet_model import *

if __name__ == '__main__':
    
    print('UNET USING GE PROBLEM EVALUATION')

    train_ids = glob.glob(os.path.join('datasets/membrane/train_aug/image', '*.png'))
    train_ids = [os.path.basename(id) for id in train_ids]
    
    dset_args = {
        "path": "datasets/membrane",
        "train_path": "datasets/membrane/train_aug",
        "test_path": "datasets/membrane/test_posproc",
        "input_shape": (256, 256, 1),
        "train_ids": train_ids,
        "valid_ids": [],
        "test_ids": [f'{i}.png' for i in range(30)],
        "train_steps": 300,
        "test_steps": 30,
        "aug": None
    }

    train_gen = DataGenerator(
        dset_args['train_path'], 
        dset_args['train_ids'], 
        dset_args['input_shape'], 
        batch_size=5,
        shuffle=False)
    test_gen = DataGenerator(
        dset_args['test_path'], 
        dset_args['test_ids'], 
        dset_args['input_shape'], 
        batch_size=5, 
        shuffle=False)

    problem = UNetProblem(None, dset_args)
    problem.train_generator = train_gen
    problem.test_generator = test_gen
    problem.verbose = True

    solution = GESolution([])
    solution.phenotype = unet(dset_args['input_shape']).to_json()

    acc = problem.evaluate(solution.phenotype)
    print(acc)
