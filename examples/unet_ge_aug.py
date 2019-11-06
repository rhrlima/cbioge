from keras.preprocessing.image import ImageDataGenerator

from algorithms.solutions import GESolution
from datasets.dataset import DataGenerator
from grammars import BNFGrammar
from problems import UNetProblem

from examples.unet_model import *

if __name__ == '__main__':
    
    dset_args = {
        "path": "datasets/membrane",
        "train_path": "datasets/membrane/train_posproc",
        "test_path": "datasets/membrane/test_posproc",
        "input_shape": (256, 256, 1),
        "train_ids": [f'{i}.png' for i in range(30)],
        "valid_ids": [],
        "test_ids": [f'{i}.png' for i in range(30)],
        "train_steps": 300,
        "test_steps": 30,
        "aug": dict(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    }

    data_aug = ImageDataGenerator(**dset_args['aug'])
    train_gen = DataGenerator(
        dset_args['train_path'], 
        dset_args['train_ids'], 
        dset_args['input_shape'], 
        batch_size=2, 
        data_aug=data_aug)
    test_gen = DataGenerator(
        dset_args['test_path'], 
        dset_args['test_ids'], 
        dset_args['input_shape'], 
        batch_size=1, 
        shuffle=False)

    problem = UNetProblem(None, dset_args)
    problem.train_generator = train_gen
    problem.test_generator = test_gen

    solution = GESolution([])
    solution.phenotype = unet(dset_args['input_shape']).to_json()

    loss, acc = problem.evaluate(solution.phenotype)
    print(loss, acc)
