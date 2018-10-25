import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import zipfile

root_folder = ''
train_file = 'fashion-mnist_train'
test_file  = 'fashion-mnist_test'
pickle_file = 'fashion-mnist.pickle'

image_size = 28
num_classes = 10

def maybe_extract(zip_file):
	zip_file = zip_file + '.csv.zip'
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

		dataset = np.asarray(dataset, dtype=np.float32)
		labels = np.asarray(labels, dtype=np.int32)
		dataset = dataset.reshape(-1, image_size, image_size)
		print('dataset ', dataset.shape)
		print('labels ', labels.shape)
		print('mean ', np.mean(dataset))
		print('std ', np.std(dataset))
	return dataset, labels


def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels


def split_dataset(dataset, labels, split_size):
	valid_dataset = dataset[:split_size,:,:]
	train_dataset = dataset[split_size:,:,:]
	valid_labels = labels[:split_size]
	train_labels = labels[split_size:]
	return train_dataset, train_labels, valid_dataset, valid_labels



maybe_extract(train_file)
maybe_extract(test_file)

train_dataset, train_labels = load_dataset_from_csv(train_file)
test_dataset, test_labels = load_dataset_from_csv(test_file)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)

train_dataset, train_labels, valid_dataset, valid_labels = split_dataset(train_dataset, train_labels, 10000)

print(train_dataset.shape, train_labels.shape)
print(valid_dataset.shape, valid_labels.shape)
print(test_dataset.shape, test_labels.shape)

pickle_file = os.path.join(root_folder, pickle_file)
with open(pickle_file, 'wb') as f:
	save = {
		'train_dataset': train_dataset, 
		'train_labels': train_labels, 
		'valid_dataset': valid_dataset, 
		'valid_labels': valid_labels, 
		'test_dataset': test_dataset, 
		'test_labels': test_labels, 
		'input_shape': (image_size, image_size, 1), 
		'num_classes': num_classes, 
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
stat_info = os.stat(pickle_file)
print('Compressed pickle size: ', stat_info.st_size)