import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

num_classes = 10
image_size = 28
root = ''
train_folder = 'notMNIST_large'
test_folder = 'notMNIST_small'

def load_dataset(folder, min_images=None, force=False):

	data_folder = os.path.join(root, folder)
	if not os.path.exists(data_folder):
		print('{} folder does not exists.'.format(data_folder))
		exit()

	data_folders = [os.path.join(data_folder, d) for d in sorted(os.listdir(data_folder)) if os.path.isdir(os.path.join(data_folder, d))]
	print(data_folders)

	pickle_files = []
	for d in data_folders:
		pickle_file = pickle_dataset(d, min_images, force)
		pickle_files.append(pickle_file)
	return pickle_files


def load_images(folder, min_images):

	pixel_depth = 255

	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), image_size, image_size))

	num_images = 0
	for image_name in image_files:
		image_file = os.path.join(folder, image_name)
		try:
			image_data = (imageio.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth

			if image_data.shape != (image_size, image_size):
				raise Exception('Unexpected image size shape' % str(image_data.shape))

			dataset[num_images] = image_data
			num_images += 1

			if min_images != None and num_images >= min_images: break
		except (IOError, ValueError) as e:
			print('Could not read: ', image_file, ':', e, 'its ok, continuing..')
	dataset = dataset[:min_images, :, :]

	print('Dataset shape: ', dataset.shape)
	print('Mean: ', np.mean(dataset))
	print('STD: ', np.std(dataset))

	return dataset


def pickle_dataset(file_path, min_images, force=False):
	file_name = file_path + '.pickle'

	if os.path.exists(file_name) and not force:
		print('Already present - skipping pickling: ', file_name)

	else:

		print('Loading images from ', file_path)
		dataset = load_images(file_path, min_images)

		print('Pickling ', file_name)
		try:
			with open(file_name, 'wb') as f:
				pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to ', file_name, ':', e)

	return file_name


def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels


def merge_datasets(pickle_files, train_size, valid_size=0):
	num_classes = len(pickle_files)
	valid_dataset, valid_labels = make_arrays(valid_size, image_size)
	train_dataset, train_labels = make_arrays(train_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes

	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):       
		try:
			with open(pickle_file, 'rb') as f:
				letter_set = pickle.load(f)
				# let's shuffle the letters to have random validation and training set
				np.random.shuffle(letter_set)
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class
										
				train_letter = letter_set[vsize_per_class:end_l, :, :]
				train_dataset[start_t:end_t, :, :] = train_letter
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise
		
	return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels


train_datasets = load_dataset(train_folder)
test_datasets = load_dataset(test_folder)


train_size = 50000#200000
valid_size = 10000
test_size = 10000


valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)


print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(root, 'notMNIST.pickle')
try:
	f = open(pickle_file, 'wb')
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
	f.close()
except Exception as e:
	print('Unable to save data to ', pickle_file, ':', e)
	raise

stat_info = os.stat(pickle_file)
print('Compressed pickle size:', stat_info.st_size)