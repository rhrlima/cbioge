import os

import numpy as np

import keras
import skimage.io as io



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

    def __init__(self, path, ids, input_shape, batch_size=32, data_aug=None, shuffle=True):
        self.path = path
        self.ids = ids
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_aug = data_aug
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
        return self._load_data_from_image(temp_ids)

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

        # loads it
        for i, id in enumerate(ids):
            x[i,] = io.imread(os.path.join(self.path, 'image', id))
            y[i,] = io.imread(os.path.join(self.path, 'label', id))

        if self.data_aug != None:
            it = self.data_aug.flow(x, y, batch_size=self.batch_size)
            return it.next()

        return x, y
