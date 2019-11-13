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
        "train_steps": 30,
        "test_steps": 30,
        "aug": dict(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')
    }

    data_aug = ImageDataGenerator(**dset_args['aug'])
    train_gen = DataGenerator(dset_args['train_path'], dset_args['input_shape'], batch_size=1, data_aug=data_aug, shuffle=True)

    for i in range(dset_args['train_steps']):
        img, msk = train_gen.__getitem__(i)
        print('---')
    #print(img.shape, img.min(), img.max(), msk.shape, msk.min(), msk.max())