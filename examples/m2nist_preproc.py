import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

import skimage.transform as trans

folder = 'datasets/m2nist'
#pickle_file = 'm2nist.pickle'
#input_shape = (64, 84, 1)
#num_classes = 1 #segmentation

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation, :, :]
    return shuffled_dataset, shuffled_labels

def split_dataset(dataset, labels, split_size):
    valid_dataset = dataset[:split_size, :, :]
    train_dataset = dataset[split_size:, :, :]
    valid_labels = labels[:split_size, :, :]
    train_labels = labels[split_size:, :, :]
    return train_dataset, train_labels, valid_dataset, valid_labels

def save(items, path):
    if not os.path.exists(os.path.join(folder, path)):
        os.makedirs(os.path.join(folder, path))

    for i, item in enumerate(items):
        np.save(os.path.join(folder, path, str(i)), item)


if __name__ == '__main__':
    
    # loads data
    data = np.load('datasets/m2nist/combined.npy')
    true = np.load('datasets/m2nist/segmented.npy')

    print('reading data')
    print(data.shape)
    print(true.shape)
    # gets just the last channel (all the classes) and invert the colors
    true = np.abs(true[:, :, :, 10] - 1)

    resized_data = np.empty((data.shape[0], 64, 64, 1))
    resized_true = np.empty((data.shape[0], 64, 64, 1))
    for i in range(data.shape[0]):
        data[i] = data[i] / 255
        true[i] = true[i] / 255
        aux1 = trans.resize(data[i], (64, 64))
        aux2 = trans.resize(true[i], (64, 64))
        aux1 = np.reshape(aux1, (64, 64, 1))
        aux2 = np.reshape(aux2, (64, 64, 1))
        resized_data[i] = aux1
        resized_true[i] = aux2

    print('resizing')
    print(resized_data.shape)
    print(resized_true.shape)

    # shuffles it
    print('shuffle')
    shuffled_data, shuffled_true = randomize(resized_data, resized_true)

    # spliting into train and validation (5000 - 500)
    print('split to train/valid')
    train_dataset, train_labels, valid_dataset, valid_labels = split_dataset(
    	shuffled_data, shuffled_true, 500)

    # spliting into train and test (4500 - 500)
    print('split to train/test')
    train_dataset, train_labels, test_dataset, test_labels = split_dataset(
    	train_dataset, train_labels, 500)

    print(train_dataset.shape, train_labels.shape)
    print(valid_dataset.shape, valid_labels.shape)
    print(test_dataset.shape, test_labels.shape)

    print('saving')
    save(train_dataset, 'train/image')
    save(train_labels, 'train/label')

    save(valid_dataset, 'valid/image')
    save(valid_labels, 'valid/label')

    save(test_dataset, 'test/image')
    save(test_labels, 'test/label')

    # pickle_file = os.path.join('.', pickle_file)
    # with open(pickle_file, 'wb') as f:
    #     save = {
    #         'train_dataset': train_dataset,
    #         'train_labels': train_labels,
    #         'valid_dataset': valid_dataset,
    #         'valid_labels': valid_labels,
    #         'test_dataset': test_dataset,
    #         'test_labels': test_labels,
    #         'input_shape': input_shape,
    #         'num_classes': num_classes
    #     }
    #     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

    # stat_info = os.stat(pickle_file)
    # print('Compressed pickle size: ', stat_info.st_size)