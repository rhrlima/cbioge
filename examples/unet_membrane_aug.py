import os

from keras import callbacks
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

from datasets.dataset import DataGenerator

import skimage.io as io

from examples.unet_model import *
from utils.image import *


if __name__ == '__main__':

    path = 'datasets/membrane'
    input_shape = (256, 256, 1)

    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    data_aug = ImageDataGenerator(**data_gen_args)

    img_ids = [f'{i}.png' for i in range(30)]

    train_gen = DataGenerator(os.path.join(path, 'train_posproc'), img_ids, input_shape, batch_size=2, data_aug=data_aug)
    test_gen = DataGenerator(os.path.join(path, 'test_posproc'), img_ids, input_shape, batch_size=1, shuffle=False)

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
