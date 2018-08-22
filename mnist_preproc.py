import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile

root_folder = 'datasets/mnist/'

train_file = 'fashion-mnist_train'
test_file  = 'fashion-mnist_test'


def maybe_extract(zip_file):
	file_path = os.path.join(root_folder, zip_file)
	with zipfile.ZipFile(file_path) as zip_ref:
		zip_ref.extractall(root_folder)
	print('extracted \"{}\" file to \"{}\" folder.'.format(zip_file, root_folder))


def load_dataset_from_csv(file, min_images=None, force=False):

	pixel_depth = 255

	file += '.csv'
	file_path = os.path.join(root_folder, file)
	with open(file_path) as f:
		f.readline() #skip headers
		
		dataset = []
		labels = []
		for i, line in enumerate(f):
			line = line.replace('\n', '').split(',')
			label = line[0]
			img_data = (np.array(line[1:], dtype=float) - pixel_depth / 2) / pixel_depth
			dataset.append(img_data)
			labels.append(label)

		dataset = np.asarray(dataset)
		labels = np.asarray(labels)
		dataset = dataset.reshape(-1, 28, 28)
		print('dataset ', dataset.shape)
		print('labels ', labels.shape)
		print('mean ', np.mean(dataset))
		print('std ', np.std(dataset))
	return dataset, labels


def pickle_dataset():
	pass


def merge_datasets():
	pass


train_dataset, train_labels = load_dataset_from_csv(train_file)
test_dataset, test_labels = load_dataset_from_csv(test_file)