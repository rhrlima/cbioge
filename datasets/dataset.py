import os

import numpy as np

import keras
import skimage.io as io

from keras.preprocessing.image import ImageDataGenerator


#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):

	'''The dataset should be organized as:
		dataset
		--train
		----image
		----label
		--validation
		----image
		----label
		--test
		----image
		----label

		in .npy files (for now)
	'''

	def __init__(self, path, ids, input_shape, batch_size=32, shuffle=True, ):
		self.path = path
		self.ids = ids
		self.input_shape = input_shape
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):

		# denotes the number of batches per epoch
		return int(np.floor(len(self.ids) / self.batch_size))

	def __getitem__(self, index):

		# generates list of random indexes
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# get the ID stored in those indexes
		temp_ids = [self.ids[i] for i in indexes]

		# load them and return
		return self._load_data(temp_ids)

	def on_epoch_end(self):

		# update indexes after each epoch
		self.indexes = np.arange(len(self.ids))

		# shuffles it when true
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def _load_data(self, ids):

		# create placeholders for data
		x = np.empty((self.batch_size, *self.input_shape))
		y = np.empty((self.batch_size, *self.input_shape))

		# loads it
		for i, id in enumerate(ids):
			x[i,] = np.load(os.path.join(self.path, 'image', id))
			y[i,] = np.load(os.path.join(self.path, 'label', id))

		#TEST
		data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

		datagen = ImageDataGenerator(**data_gen_args)
		datagen.flow(x, y, batch_size=self.batch_size)

		return datagen.next()

	# def _load_data_from_image(self, ids):

	# 	# create placeholders for data
	# 	x = np.empty((self.batch_size, 256, 256, 1))
	# 	y = np.empty((self.batch_size, 256, 256, 1))

	# 	# loads it
	# 	for i, id in enumerate(ids):
	# 		print(os.path.join(self.path, 'image', id))
	# 		x[i,] = io.imread(os.path.join(self.path, 'image', id), as_gray = True)
	# 		y[i,] = io.imread(os.path.join(self.path, 'label', id), as_gray = True)

	# 	return x, y