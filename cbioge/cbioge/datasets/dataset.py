import os
import glob
import pickle

import numpy as np

import keras
from keras.utils import np_utils

# import skimage.io as io

# from cbioge.utils.image import *

# # TODO sem uso, util pra "criar" dataset a partir de uma pasta com imagens
# class DataGenerator(keras.utils.Sequence):

#     '''The dataset should be organized as:
#         --path/
#         ----image/
#         ------file0
#         ------file1
#         ----label/
#         ------file0
#         ------file1
#     '''
#     def __init__(self, path, input_shape, batch_size=32, data_aug=None, shuffle=True):
#         self.path = path
#         self.input_shape = input_shape
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.data_aug = data_aug

#         self.ids = self._get_ids()
#         self.on_epoch_end()

#     def __len__(self):

#         # denotes the number of batches per epoch
#         return int(np.floor(len(self.ids) / self.batch_size))

#     def __getitem__(self, index):

#         # generates list of random indexes
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # get the ID stored in those indexes
#         temp_ids = [self.ids[i] for i in indexes]

#         # load them and return
#         if self.npy:
#             return self._load_data_from_npy(temp_ids)
#         return self._load_data_from_image(temp_ids)

#     def _get_ids(self):        
#         # list and get name of files inside path
#         complete_path = os.path.join(self.path, 'image', '*')
#         files = glob.glob(complete_path)

#         if len(files) == 0:
#             raise NameError('No files found in', complete_path)

#         self.npy = True if os.path.splitext(files[0])[1] == '.npy' else False
        
#         print(f'{len(files):6} files found, NPY', self.npy)

#         return [os.path.basename(id) for id in files]

#     def on_epoch_end(self):

#         # update indexes after each epoch
#         self.indexes = np.arange(len(self.ids))

#         # shuffles it when true
#         if self.shuffle:
#             np.random.shuffle(self.indexes)

#     def _load_data_from_npy(self, ids):

#         # create placeholders for data
#         x = np.empty((self.batch_size, *self.input_shape))
#         y = np.empty((self.batch_size, *self.input_shape))

#         # loads it
#         for i, id in enumerate(ids):
#             x[i,] = np.load(os.path.join(self.path, 'image', id))
#             y[i,] = np.load(os.path.join(self.path, 'label', id))

#         if self.data_aug != None:
#             it = self.data_aug.flow(x, y, batch_size=self.batch_size)
#             return it.next()

#         return x, y

#     def _load_data_from_image(self, ids):

#         # create placeholders for data
#         x = np.empty((self.batch_size, *self.input_shape))
#         y = np.empty((self.batch_size, *self.input_shape))

#         for i, id in enumerate(ids):
            
#             # read image and label from source
#             img = io.imread(os.path.join(self.path, 'image', id), as_gray=True)
#             msk = io.imread(os.path.join(self.path, 'label', id), as_gray=True)

#             #print(id, img.shape, msk.shape, img.min(), img.max(), 'loaded')

#             # reshape to (w, h, c, 1)
#             x[i,] = np.reshape(img, img.shape+(1,))
#             y[i,] = np.reshape(msk, msk.shape+(1,))

#         # create augmentation if applicable
#         if self.data_aug != None:
#             it = self.data_aug.flow(x, y, batch_size=self.batch_size, shuffle=self.shuffle)
#             x, y = it.next()

#         for i in range(self.batch_size):
#             # normalize
#             img = normalize(x[i])
#             msk = normalize(y[i])

#             # binarize mask
#             msk = binarize(msk)

#             x[i,] = img
#             y[i,] = msk

#             #print(i, x[i].shape, y[i].shape, x[i].min(), x[i].max(), 'processed')

#         return x, y


class Dataset():

    def __init__(self, data_dict, **kwargs):
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
