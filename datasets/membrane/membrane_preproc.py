import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import skimage.io as io
import skimage.transform as trans

from keras.preprocessing.image import ImageDataGenerator

from utils.image import *


base_path = 'datasets/membrane'


def load_images(path, mask=True):
    images = []
    files = glob.glob(os.path.join(base_path, path, '*.png'))
    for f in files:
        img = io.imread(f, as_gray=True)
        img = trans.resize(img, (256, 256))
        img = normalize(img)
        if mask:
            img = binarize(img)
        img = np.reshape(img, img.shape+(1,))
        print(f, img.shape, img.min(), img.max())
        images.append(img)
    return images


def save_images(path, items, npy=False, force=False):
    complete_path = os.path.join(base_path, path)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    for i, item in enumerate(items):
        file_type = '.npy' if npy else '.png'
        file_name = os.path.join(complete_path, str(i)+file_type)
        print('saving', file_name)
        if not os.path.exists(file_name) or force:
            print('saving', file_name)
            if npy:
                np.save(file_name, item)
            else:
                io.imsave(file_name, item)


def generate_augmented_data(batch_size, path, aug_dict, target_size = (256, 256), seed = 1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_gen = image_datagen.flow_from_directory(
        path,
        classes = ['image'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = os.path.join(base_path, path, 'image'),
        save_prefix  = '',
        seed = seed)
    mask_gen = mask_datagen.flow_from_directory(
        path,
        classes = ['label'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = os.path.join(base_path, path, 'label'),
        save_prefix  = '',
        seed = seed)

    for img, mask in zip(image_gen, mask_gen):
        if np.max(img) > 1:
            img = normalize(img)
            mask = normalize(mask)
            mask = binarize(mask)
        yield img, mask


if __name__ == '__main__':

    images = load_images('train/image', False)
    masks = load_images('train/label')
    save_images('train_posproc/image', images)
    save_images('train_posproc/label', masks)
    #save_images('npy/train/image', images, npy=True)
    #save_images('npy/train/label', masks, npy=True)

    images = load_images('test/image', False)
    masks = load_images('test/label')
    save_images('test_posproc/image', images)
    save_images('test_posproc/label', masks)
    #save_images('npy/test/image', images, npy=True)
    #save_images('npy/test/label', masks, npy=True)

    data_gen_args = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')

    if not os.path.exists('train_aug/image'):
        os.makedirs('train_aug/image')
        os.makedirs('train_aug/label')

    #aug_gen = generate_augmented_data(1, 'train_aug', data_gen_args)

    #for _ in range(600):
    #    next(aug_gen)