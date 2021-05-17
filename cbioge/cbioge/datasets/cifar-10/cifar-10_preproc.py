#https://www.cs.toronto.edu/~kriz/cifar.html

import os

import pickle
import numpy as np

import matplotlib.pyplot as plt

pickle_file = 'cifar-10.pickle'

image_size = 32
channels = 3

INPUT_SHAPE = (image_size, image_size, channels)
NUM_CLASSES = 10


def load_dataset_from_file(file, min_images=None, force=False):
	'''loads the data from a bytes file that returns a dict
		normalizes the image pixel values to [0, 1] and
		reshapes the pixels to a 32x32x3 shape
	'''
	pixel_depth = 255

	with open(file, 'rb') as f:
		data = pickle.load(f, encoding='bytes')

	dataset = data[b'data'] / pixel_depth
	dataset = dataset.reshape(-1, image_size, image_size, channels, order='F')
	#dataset = rgb2grey(dataset)

	labels = np.asarray(data[b'labels'], dtype=np.int32)
	print('dataset ', dataset.shape)
	print('labels ', labels.shape)
	print('mean ', np.mean(dataset))
	print('std ', np.std(dataset))
	
	return dataset, labels


def split_dataset(dataset, labels, split_size):
	valid_dataset = dataset[:split_size,:,:]
	train_dataset = dataset[split_size:,:,:]
	valid_labels = labels[:split_size]
	train_labels = labels[split_size:]
	return train_dataset, train_labels, valid_dataset, valid_labels


def rgb2grey(dataset):
	r = dataset[:, :, :, 0]
	g = dataset[:, :, :, 1]
	b = dataset[:, :, :, 2]
	return 0.21 * r + 0.72 * g + 0.07 * b


if __name__ == '__main__':

	dataset = np.ndarray(shape=(0, image_size, image_size, channels))
	labels = np.ndarray(shape=(0,))

	for i in range(5):
		d, l = load_dataset_from_file('data_batch_{}'.format(i+1))
		dataset = np.concatenate((dataset, d))
		labels = np.concatenate((labels, l))

	test_dataset, test_labels = load_dataset_from_file('test_batch')

	train_dataset, train_labels, valid_dataset, valid_labels = split_dataset(
		dataset, labels, 10000)

	print('train', train_dataset.shape, train_labels.shape)
	print('valid', valid_dataset.shape, valid_labels.shape)
	print('test ', test_dataset.shape, test_labels.shape)

	with open(pickle_file, 'wb') as f:
		save = {
			'train_dataset': train_dataset, 
			'train_labels': train_labels, 
			'valid_dataset': valid_dataset, 
			'valid_labels': valid_labels, 
			'test_dataset': test_dataset, 
			'test_labels': test_labels, 
			'input_shape': INPUT_SHAPE, 
			'num_classes': NUM_CLASSES
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

	stat_info = os.stat(pickle_file)
	print('Compressed pickle size: ', stat_info.st_size)