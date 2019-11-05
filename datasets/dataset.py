import os
import glob

import numpy as np

import keras
import skimage.io as io

from utils.image import *

class DataGenerator(keras.utils.Sequence):

    '''https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    The dataset should be organized as:
        dataset/
        --train/
        ----image/
        ----label/
        --test/
        ----image/
        ----label/
    '''

    def __init__(self, path, input_shape, batch_size=32, data_aug=None, shuffle=True, npy=False):
        self.path = path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_aug = data_aug
        self.npy = npy
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
        file_type = '*.png' if not self.npy else '*.npy'
        files = glob.glob(os.path.join(self.path, 'image', file_type))
        file_names = [os.path.basename(id) for id in files]
        print(f'{len(file_names):6} files found')
        return file_names

    def on_epoch_end(self):

        # update indexes after each epoch
        self.indexes = np.arange(len(self.ids))

        # shuffles it when true
        if self.shuffle:
            np.random.shuffle(self.indexes)

    #CHECK
    def _post_process(self, img, msk):
        # normalize
        img = normalize(img)
        msk = normalize(msk)

        # binarize mask
        msk = binarize(msk)

        # reshape to (w, h, 1)
        img = np.reshape(img, img.shape+(1,))
        msk = np.reshape(msk, msk.shape+(1,))

        return img, msk

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

            # normalize
            img = normalize(img)
            msk = normalize(msk)

            # binarize mask
            msk = binarize(msk)

            # reshape to (w, h, 1)
            x[i,] = np.reshape(img, img.shape+(1,))
            y[i,] = np.reshape(msk, msk.shape+(1,))

            # print(id, x[i].shape, y[i].shape, x[i].min(), x[i].max())

        if self.data_aug != None:
            it = self.data_aug.flow(x, y, batch_size=self.batch_size)
            return it.next()

        return x, y
