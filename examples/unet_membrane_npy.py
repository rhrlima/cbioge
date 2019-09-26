import os

import numpy as np

from keras import callbacks
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

from datasets.dataset import DataGenerator

import skimage.io as io
import skimage.transform as trans

from unet_model import *
from utils.image import *


if __name__ == '__main__':

    input_shape = (256, 256, 1)

    train_ids = [f'{i}.npy' for i in range(4000)]
    test_ids = train_ids[:500]

    train_gen = DataGenerator(os.path.join(path, 'npy/train'), train_ids, input_shape, batch_size=2)
    test_gen = DataGenerator(os.path.join(path, 'npy/test'), test_ids, input_shape, batch_size=1, shuffle=False)

    model = unet(input_size=input_shape)
    model.fit_generator(train_gen, steps_per_epoch=300, epochs=1, verbose=1, use_multiprocessing=True, workers=4)
    
    results = model.predict_generator(test_gen, 30, verbose=1)

    acc = 0.0
    for i, pred in enumerate(results):
        io.imsave(f'datasets/membrane/npy/test/pred/{i}.png', pred)
        true = io.imread(f'datasets/membrane/test/label/{i}.png')
        pred = adjust_image(pred)
        true = adjust_image(true)
        acc += iou_accuracy(true, pred)

    print('acc', acc/len(results))
