import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import skimage.io as io
import skimage.transform as trans

from keras.preprocessing.image import ImageDataGenerator

def load_images(path, mask=True):
    images = []
    files = glob.glob(os.path.join(path, '*.png'))
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

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

def binarize(mask, threshold=0.5):
    mask[mask > threshold ] = 1
    mask[mask <= threshold] = 0
    return mask

def save_to_npy(path, items, force=True):

    if not os.path.exists(path):
        os.makedirs(path)

    for i, item in enumerate(items):
        file_name = os.path.join(path, f'{i}.npy')
        if not os.path.exists(file_name) or force:
            print('saving', file_name)
            np.save(file_name, item)

def save_to_images(path, items, force=True):

    if not os.path.exists(path):
        os.makedirs(path)

    for i, item in enumerate(items):
        file_name = os.path.join(path, f'{i}.png')
        if not os.path.exists(file_name) or force:
            print('saving', file_name)
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
        save_to_dir = os.path.join(path, 'aug/image'),
        save_prefix  = '',
        seed = seed)
    mask_gen = mask_datagen.flow_from_directory(
        path,
        classes = ['label'],
        class_mode = None,
        color_mode = "grayscale",
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = os.path.join(path, 'aug/label'),
        save_prefix  = '',
        seed = seed)

    for img, mask in zip(image_gen, mask_gen):
        if np.max(img) > 1:
            img = normalize(img)
            mask = normalize(mask)
            mask = binarize(mask)
        print(img.shape, mask.shape)
        yield img, mask

if __name__ == '__main__':

    images = load_images('train/image', False)
    masks = load_images('train/label')
    #save_to_images('posproc/train/image', images)
    #save_to_images('posproc/train/label', masks)
    #save_to_npy('npy/train/image', images)
    #save_to_npy('npy/train/label', masks)

    images = load_images('test/image', False)
    masks = load_images('test/label')
    #save_to_images('posproc/test/image', images)
    #save_to_images('posproc/test/label', masks)
    #save_to_npy('npy/test/image', images)
    #save_to_npy('npy/test/label', masks)

    data_gen_args = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')

    if not os.path.exists('train/aug/image'):
        os.makedirs('train/aug/image')
        os.makedirs('train/aug/label')

    aug_gen = generate_augmented_data(1, 'train', data_gen_args)

    for _ in range(600):
        next(aug_gen)