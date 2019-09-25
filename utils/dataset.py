import glob
import os

import numpy

import skimage.io as io

def read_dataset_from_directory(path):
	'''expects two folders inside path:
		dataset/
		--image/
		--label/
	'''
	images = glob.glob(os.path.join(path, 'image', '*'))
	labels = glob.glob(os.path.join(path, 'label', '*'))

	for img, msk in zip(images, labels):

		img = io.imread(img)
		msk = io.imread(msk)

		print(img.shape, msk.shape)