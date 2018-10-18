import numpy as np
import pickle

import matplotlib.pyplot as plt

files = [
	'data_batch_1',
	#'data_batch_2',
	#'data_batch_3',
	#'data_batch_4',
	#'data_batch_5',
	#'test_batch',
]

labels = [
	'airplane',
	'automobile',
	'bird',
	'cat',
	'deer',
	'dog',
	'frog',
	'horse',
	'ship',
	'truck',
]

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

train

for file in files:

	with open(file, 'rb') as f:
		data = pickle.load(f, encoding='bytes')

	print(data.keys())
	print(file)
	print('samples', len(data[b'filenames']))

	# sample
	for index in range(100):
		#index = 102
		print('filename', data[b'filenames'][index])
		print('label', data[b'labels'][index], labels[data[b'labels'][index]])

		img = data[b'data'][index].reshape((32, 32, 3), order='F')

	#plt.imshow(img)
	#plt.show()