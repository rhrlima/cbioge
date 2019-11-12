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


def load_images(path, input_shape, num_images=0):
    files = glob.glob(os.path.join(path, 'image', '*'))
    file_names = [os.path.basename(id) for id in files]

    num_images = min(num_images, len(files))

    print('found', len(files), 'images')

    x = np.empty((num_images, *input_shape))
    y = np.empty((num_images, *input_shape))

    for i in range(num_images):
        img = io.imread(os.path.join(path, 'image', file_names[i]), as_gray=True)
        mask = io.imread(os.path.join(path, 'label', file_names[i]), as_gray=True)

        img = normalize(img)
        mask = normalize(mask)
        mask = binarize(mask)

        img = np.reshape(img, img.shape+(1,))
        mask = np.reshape(mask, mask.shape+(1,))

        x[i,] = img
        y[i,] = mask

    return x, y


if __name__ == '__main__':

    args = get_args()

    print(args)

    dataset = json.loads(open(args.dataset, 'r').read())
    if not args.train is None:
        dataset['train_steps'] = args.train
    if not args.test is None:
        dataset['test_steps'] = args.test

    x_train, y_train = load_images(dataset['train_path'], dataset['input_shape'], dataset['train_steps'])
    x_test, y_test = load_images(dataset['test_path'], dataset['input_shape'], dataset['test_steps'])

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # model = unet(dataset['input_shape'])

    # model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.fit(x_train, y_train, batch_size=args.batch, epochs=args.epochs, verbose=args.verbose)
    # loss, acc = model.evaluate(x_test, y_test, batch_size=args.batch, verbose=args.verbose)
    # print('loss', loss, 'acc', acc)
