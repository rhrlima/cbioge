import os
import glob
import pickle

import numpy as np

import keras
import skimage.io as io

from cbioge.utils.image import *

# TODO sem uso, util pra "criar" dataset a partir de uma pasta com imagens
class DataGenerator(keras.utils.Sequence):

    '''The dataset should be organized as:
        --path/
        ----image/
        ------file0
        ------file1
        ----label/
        ------file0
        ------file1
    '''

    def __init__(self, path, input_shape, batch_size=32, data_aug=None, shuffle=True):
        self.path = path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_aug = data_aug

        self.ids = self._get_ids()
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
        if self.npy:
            return self._load_data_from_npy(temp_ids)
        return self._load_data_from_image(temp_ids)

    def _get_ids(self):        
        # list and get name of files inside path
        complete_path = os.path.join(self.path, 'image', '*')
        files = glob.glob(complete_path)

        if len(files) == 0:
            raise NameError('No files found in', complete_path)

        self.npy = True if os.path.splitext(files[0])[1] == '.npy' else False
        
        print(f'{len(files):6} files found, NPY', self.npy)

        return [os.path.basename(id) for id in files]

    def on_epoch_end(self):

        # update indexes after each epoch
        self.indexes = np.arange(len(self.ids))

        # shuffles it when true
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_data_from_npy(self, ids):

        # create placeholders for data
        x = np.empty((self.batch_size, *self.input_shape))
        y = np.empty((self.batch_size, *self.input_shape))

        # loads it
        for i, id in enumerate(ids):
            x[i,] = np.load(os.path.join(self.path, 'image', id))
            y[i,] = np.load(os.path.join(self.path, 'label', id))

        if self.data_aug != None:
            it = self.data_aug.flow(x, y, batch_size=self.batch_size)
            return it.next()

        return x, y

    def _load_data_from_image(self, ids):

        # create placeholders for data
        x = np.empty((self.batch_size, *self.input_shape))
        y = np.empty((self.batch_size, *self.input_shape))

        for i, id in enumerate(ids):
            
            # read image and label from source
            img = io.imread(os.path.join(self.path, 'image', id), as_gray=True)
            msk = io.imread(os.path.join(self.path, 'label', id), as_gray=True)

            #print(id, img.shape, msk.shape, img.min(), img.max(), 'loaded')

            # reshape to (w, h, c, 1)
            x[i,] = np.reshape(img, img.shape+(1,))
            y[i,] = np.reshape(msk, msk.shape+(1,))

        # create augmentation if applicable
        if self.data_aug != None:
            it = self.data_aug.flow(x, y, batch_size=self.batch_size, shuffle=self.shuffle)
            x, y = it.next()

        for i in range(self.batch_size):
            # normalize
            img = normalize(x[i])
            msk = normalize(y[i])

            # binarize mask
            msk = binarize(msk)

            x[i,] = img
            y[i,] = msk

            #print(i, x[i].shape, y[i].shape, x[i].min(), x[i].max(), 'processed')

        return x, y


def read_dataset_from_pickle(pickle_file):
    ''' Reads a dataset stored in a pickle file. Expects a pickle file 
        containing a dict structure.
    '''
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    return data


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
