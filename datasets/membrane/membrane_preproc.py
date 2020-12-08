import glob
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

import skimage.io as io
import skimage.transform as trans

from keras.preprocessing.image import ImageDataGenerator

from utils.image import *

pickle_file = 'membrane.pickle'
img_shape = (256, 256)
base_path = 'datasets/membrane'


def load_images(path, mask=False, log=False):
    files = glob.glob(os.path.join(base_path, path, '*.png'))
    images = np.empty((len(files), *img_shape, 1))
    for i, f in enumerate(files):
        img = io.imread(f, as_gray=True)
        img = trans.resize(img, img_shape)
        img = normalize(img)
        if mask:
            img = binarize(img)
        img = np.reshape(img, img.shape+(1,))
        if log: print(f, img.shape, img.min(), img.max())
        images[i,] = img
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


def generate_augmented_data(x, y, amount, seed=1):

    aug_dict = dict(rotation_range=0.2,
                width_shift_range=0.05,
                height_shift_range=0.05,
                shear_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest')

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_gen = image_datagen.flow(x, batch_size=1, seed=seed)
    mask_gen = image_datagen.flow(y, batch_size=1, seed=seed)

    x_aug = np.empty((amount, *img_shape, 1))
    y_aug = np.empty((amount, *img_shape, 1))
    for i in range(amount):
        img = next(image_gen)
        mask = next(mask_gen)

        x_aug[i,] = img[0]
        y_aug[i,] = mask[0]

        # plt.subplot(1, 2, 1)
        # plt.imshow(img[0,:,:,0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask[0,:,:,0])
        # plt.show()
    return x_aug, y_aug


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation, :, :]
    return shuffled_dataset, shuffled_labels


def split_dataset(dataset, labels, split_size):
    valid_dataset = dataset[:split_size,:,:]
    train_dataset = dataset[split_size:,:,:]
    valid_labels = labels[:split_size]
    train_labels = labels[split_size:]
    return train_dataset, train_labels, valid_dataset, valid_labels


if __name__ == '__main__':

    train_dataset = load_images('train/image')
    train_labels = load_images('train/label', mask=True)
    #save_images('train_posproc/image', images)
    #save_images('train_posproc/label', masks)
    #save_images('npy/train/image', images, npy=True)
    #save_images('npy/train/label', masks, npy=True)

    print('original')
    print(train_dataset.shape, train_labels.shape)

    test_dataset = load_images('test/image')
    test_labels = load_images('test/label', mask=True)
    #save_images('test_posproc/image', images)
    #save_images('test_posproc/label', masks)
    #save_images('npy/test/image', images, npy=True)
    #save_images('npy/test/label', masks, npy=True)
    print(test_dataset.shape, test_dataset.shape)

    train_dataset, train_labels = generate_augmented_data(train_dataset, train_labels, amount=330)
    print('augmented')
    print(train_dataset.shape, train_labels.shape)

    train_dataset, train_labels, valid_dataset, valid_labels = split_dataset(train_dataset, train_labels, 30)
    print('split')
    print(train_dataset.shape, train_labels.shape)
    print(valid_dataset.shape, valid_labels.shape)



    # with open(pickle_file, 'wb') as f:
    #     save = {
    #         'train_dataset': train_dataset, 
    #         'train_labels': train_labels, 
    #         'valid_dataset': valid_dataset, 
    #         'valid_labels': valid_labels, 
    #         'test_dataset': test_dataset, 
    #         'test_labels': test_labels, 
    #         'input_shape': train_dataset[0].shape
    #     }
    #     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

    # stat_info = os.stat(pickle_file)
    # print('Compressed pickle size: ', stat_info.st_size)