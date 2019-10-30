import os

import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from examples.unet_model import *
from utils.image import *


def train_generator(train_path, batch_size, aug_dict, target_size = (256, 256)):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = ['image'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = 'image',
        seed = 1)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = ['label'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = 'mask',
        seed = 1)

    for img, mask in zip(image_generator, mask_generator):
        img = normalize(img)
        mask = normalize(mask)
        mask = binarize(mask)
        yield img, mask


def test_generator(test_path, num_image = 30, target_size = (256, 256)):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, 'image', f'{i}.png'), as_gray = True)
        msk = io.imread(os.path.join(test_path, 'label', f'{i}.png'), as_gray = True)
        img = trans.resize(img, target_size)
        img = normalize(img)
        msk = normalize(msk)
        msk = binarize(msk)
        img = np.reshape(img, img.shape+(1,)) # (256, 256, 1)
        msk = np.reshape(msk, msk.shape+(1,)) # (256, 256, 1)
        img = np.reshape(img,(1,)+img.shape) # (1, 256, 256, 1)
        msk = np.reshape(msk,(1,)+msk.shape) # (1, 256, 256, 1)
        yield img, msk


if __name__ == '__main__':

    input_shape = (256, 256, 1)

    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    train_gen = train_generator('datasets/membrane/train', 2, aug_dict=data_gen_args)
    test_gen = test_generator('datasets/membrane/test')

    model = unet(input_shape)
    model.fit_generator(train_gen, steps_per_epoch=300, epochs=1, verbose=1)
    
    loss, acc = model.evaluate_generator(test_gen, steps=30, verbose=1)
    print('loss', loss, 'acc', acc)
