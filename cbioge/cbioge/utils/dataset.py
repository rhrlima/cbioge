import glob
import os
import sys

import numpy as np

import skimage.io as io

def read_dataset_from_directory(path, npy=False):
	'''expects two folders inside path:
		dataset/
		--image/
		--label/
	'''
	images = glob.glob(os.path.join(path, 'image', '*'))
	labels = glob.glob(os.path.join(path, 'label', '*'))

	for img, msk in zip(images, labels):

		img = np.load(img) if npy else io.imread(img)
		msk = np.load(msk) if npy else io.imread(msk)

		print(img.shape, img.min(), img.max(), msk.shape, msk.min(), msk.max())


def read_dataset_from_pickle():
	pass

def read_dataset_from_dict():
	pass

def split_dataset(data, labels, split_size):
	'''splits the array into two arrays of data

	fist returned array has the split_size, second 
	has the remainder of content

	split size: number of images
	'''
	data_a = data[:split_size,:,:]
	data_b = data[split_size:,:,:]
	label_a = labels[:split_size]
	label_b = labels[split_size:]
	return data_a, label_a, data_b, label_b
