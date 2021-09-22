import pickle
import numpy as np
from keras.utils import np_utils


class Dataset:
    '''Dataset class that holds the most common data structures used in
    the training, validation and test of deep neural networks.'''

    def __init__(self, data_dict, **kwargs):
        '''# Parameters
        - x_train: train data
        - y_train: train labels
        - x_test: test data
        - y_test: test labels

        ## Optional
        - x_valid: validation data
        - y_valid: validation labels
        - num_classes: number of classes (if any)
        - train_size: defines the number of training instances (default len(x_train))
        - test_size: defines the number of test instances (default len(x_test))
        - valid_size: defines the number of validation instances 
        (default len(x_valid) if exists)
        - valid_split: float between [0, 1] that expresses the % of the training
        data that will be used as validation'''

        self.input_shape = data_dict['input_shape']

        self.x_train = data_dict['x_train']
        self.y_train = data_dict['y_train']

        self.x_test = data_dict['x_test']
        self.y_test = data_dict['y_test']

        if 'train_size' in kwargs:
            self.train_size = kwargs['train_size']
        else:
            self.train_size = len(self.x_train)

        if 'test_size' in kwargs:
            self.test_size = kwargs['test_size']
        else:
            self.test_size = len(self.x_test)

        # adds a valid set if data is present in dict
        if 'x_valid' in data_dict and 'y_valid' in data_dict:
            self.x_valid = data_dict['x_valid']
            self.y_valid = data_dict['y_valid']
            self.valid_size = len(self.x_valid)

        # creates a valid set using a portion of the training set
        elif 'valid_split' in kwargs and 'valid_size' not in kwargs:
            kwargs['valid_size'] = int(self.train_size * kwargs['valid_split'])
            self.train_size -= kwargs['valid_size']
            #kwargs.pop('valid_split')

        # creates a valid set using a sample of the training set
        if 'valid_size' in kwargs:
            self.valid_size = kwargs['valid_size']
            self.x_valid, self.y_valid, self.x_train, self.y_train = self.split(
                self.x_train, self.y_train, self.valid_size)
            #kwargs.pop('valid_size')

        # adds the number of classes if key exists in dict
        # reshapes labels to be categorical
        if 'num_classes' in data_dict:
            self.num_classes = data_dict['num_classes']
            self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
            self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)
            if hasattr(self, 'y_valid'):
                self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)

    @classmethod
    def from_pickle(cls, pickle_file, **kwargs):
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f)
        return cls(data_dict, **kwargs)

    def split(self, data, labels, split_size=None, split=None):
        '''Splits the array into two arrays of data.

        # Parameters
        data: list of data
        labels: list of labels
        split_size: integer value related to the split size

        # Return
        two sets of data grouped as (data_a, labels_a), (data_b, labelb_b)\n
        data_a and labels_a have len() of split_size\n
        data_b and labels_b have the remainder of data\n
        '''
        data_a = data[:split_size]
        data_b = data[split_size:]
        label_a = labels[:split_size]
        label_b = labels[split_size:]
        return data_a, label_a, data_b, label_b

    def shuffle(self, data, labels):
        '''Shuffles a pair of data and labels to keep consistency
        '''
        indexes = np.random.permutation(np.arange(len(data)))
        return data[indexes], labels[indexes]

    def get_data(self, set_name, sample_size=None, shuffle=False):
        if set_name == 'train':
            x_data = self.x_train
            y_data = self.y_train
            d_size = self.train_size
        elif set_name == 'valid':
            x_data = self.x_valid
            y_data = self.y_valid
            d_size = self.valid_size
        elif set_name == 'test':
            x_data = self.x_test
            y_data = self.y_test
            d_size = self.test_size

        if sample_size is not None:
            d_size = sample_size

        if shuffle:
            x_data, y_data = self.shuffle(x_data, y_data)

        return x_data[:d_size], y_data[:d_size]


def read_dataset_from_pickle(pickle_file):
    ''' Reads a dataset stored in a pickle file. Expects a pickle file 
        containing a dict structure.
    '''
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    return data


def read_dataset_from_npy(npy_file):
    pass


def read_dataset_from_directory(path):
    ''' Reads a path and loads the content into a dict. Expects two folders 
        inside path:
        dataset/
        --image/
        --label/

        return: dict containing the keys expected by the Problem class
    '''
    # images = glob.glob(os.path.join(path, 'image', '*'))
    # labels = glob.glob(os.path.join(path, 'label', '*'))

    # for img, msk in zip(images, labels):

    # 	img = np.load(img) if npy else io.imread(img)
    # 	msk = np.load(msk) if npy else io.imread(msk)

    #     print(img.shape, img.min(), img.max(), msk.shape, msk.min(), msk.max())

    return {}


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
