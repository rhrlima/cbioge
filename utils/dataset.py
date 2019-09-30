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


if __name__ == '__main__':

	if len(sys.argv) == 2:
		read_dataset_from_directory(sys.argv[1])
	elif len(sys.argv) == 3:
		read_dataset_from_directory(sys.argv[1], sys.argv[2])
	else:
		print('expected: script.py <path> [True|False]')