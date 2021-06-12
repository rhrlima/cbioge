import pickle


def read_dataset_from_pickle(pickle_file):
    ''' Reads a dataset stored in a pickle file. Expects a pickle file 
        containing a dict structure.
    '''
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    return data


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
