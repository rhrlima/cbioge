import glob
import os

from keras.models import *
from keras.layers import *
from keras.optimizers import *

from datasets.dataset import DataGenerator

import skimage.io as io

from examples.unet_model import *
from utils.image import *


if __name__ == '__main__':

    path = 'datasets/membrane'
    input_shape = (256, 256, 1)

    train_ids = glob.glob(os.path.join(path, 'train_aug/image', '*.png'))
    train_ids = [os.path.basename(id) for id in train_ids]
    test_ids = [f'{i}.png' for i in range(30)]

    train_gen = DataGenerator(os.path.join(path, 'train_aug'), train_ids, input_shape, batch_size=2)
    test_gen = DataGenerator(os.path.join(path, 'test_posproc'), test_ids, input_shape, batch_size=1, shuffle=False)

    model = unet(input_shape)
    model.fit_generator(train_gen, steps_per_epoch=300, epochs=1, verbose=1)

    loss, acc = model.evaluate_generator(test_gen, steps=30, verbose=1)
    print('loss', loss, 'acc', acc)

    # results = model.predict_generator(test_gen, 30, verbose=1)

    # acc = 0.0
    # for i, pred in enumerate(results):
    #     io.imsave(f'datasets/membrane/test/pred/{i}.png', pred)
        
    #     true = io.imread(f'datasets/membrane/test/label/{i}.png', as_gray=True)
    #     pred = normalize(pred)
    #     pred = binarize(pred)
    #     true = normalize(true)
    #     true = binarize(true)
    #     acc += iou_accuracy(true, pred)

    # print('acc', acc/len(results))
