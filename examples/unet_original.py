import argparse
import glob
import json
import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from examples.unet_model import *
from utils.image import *


def get_args():

    args = argparse.ArgumentParser(prog='script.py')

    args.add_argument('dataset', type=str) #dataset

    args.add_argument('-trs', '--train', type=int, default=None) #train steps
    args.add_argument('-tes', '--test', type=int, default=None) #test steps
    args.add_argument('-b', '--batch', type=int, default=1) #batch
    args.add_argument('-e', '--epochs', type=int, default=1) #epochs
    args.add_argument('-v', '--verbose', type=int, default=1) #verbose

    return args.parse_args()


def train_generator(train_path, batch_size, aug_dict, target_size):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = ['image'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = (256, 256),
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = 'image',
        seed = 1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = ['label'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = (256, 256),
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = 'mask',
        seed = 1)

    for img, mask in zip(image_generator, mask_generator):
        img = normalize(img)
        mask = normalize(mask)
        mask = binarize(mask)
        yield img, mask


def test_generator(test_path, target_size):
    files = glob.glob(os.path.join(test_path, 'image', '*'))
    file_names = [os.path.basename(id) for id in files]
    for file in file_names:
        img = io.imread(os.path.join(test_path, 'image', file), as_gray=True)
        msk = io.imread(os.path.join(test_path, 'label', file), as_gray=True)
        img = normalize(img)
        msk = normalize(msk)
        msk = binarize(msk)
        img = np.reshape(img, (1,)+img.shape+(1,)) # (256, 256, 1)
        msk = np.reshape(msk, (1,)+msk.shape+(1,)) # (256, 256, 1)
        #img = np.reshape(img,(1,)+img.shape) # (1, 256, 256, 1)
        #msk = np.reshape(msk,(1,)+msk.shape) # (1, 256, 256, 1)
        yield img, msk


if __name__ == '__main__':

    args = get_args()
    print(args)

    dataset = json.loads(open(args.dataset, 'r').read())
    if not args.train is None:
        dataset['train_steps'] = args.train
    if not args.test is None:
        dataset['test_steps'] = args.test

    train_gen = train_generator(dataset['train_path'], args.batch, dataset['aug'], dataset['input_shape'])
    test_gen = test_generator(dataset['test_path'], dataset['input_shape'])

    model = unet(dataset['input_shape'])

    model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit_generator(train_gen, steps_per_epoch=dataset['train_steps'], epochs=args.epochs, verbose=args.verbose)
    
    loss, acc = model.evaluate_generator(test_gen, steps=dataset['test_steps'], verbose=args.verbose)

    print('loss', loss, 'acc', acc)
