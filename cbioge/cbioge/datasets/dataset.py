import pickle

import numpy as np

from keras.utils import np_utils


class Dataset:
    '''Dataset class that holds the most common data structures used in
    the training, validation and test of deep neural networks.'''

    def __init__(self,
        x_train: list,
        y_train: list,
        x_test: list,
        y_test: list,
        x_valid: list=None,
        y_valid: list=None,
        input_shape: tuple=None,
        num_classes: int=None,
        train_size: int=None,
        test_size: int=None,
        valid_size: int=None,
        valid_split: float=None
    ):
        '''# Parameters
        - x_train: train data
        - y_train: train labels
        - x_test: test data
        - y_test: test labels

        # Optional
        - x_valid: validation data
        - y_valid: validation labels
        - input_shape: shape of input data
        - num_classes: number of classes (if any)
        - train_size: defines the number of training instances (default len(x_train))
        - test_size: defines the number of test instances (default len(x_test))
        - valid_size: defines the number of validation instances
        (default len(x_valid) if exists). It has priotity over split_size
        - valid_split: float between [0, 1] that expresses the % of the training
        data that will be used as validation'''

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

        self.x_valid = x_valid
        self.y_valid = y_valid

        self.train_size = self._parse_attr_size(train_size, x_train)
        self.test_size = self._parse_attr_size(test_size, x_test)
        self.valid_size = valid_size

        self.input_shape = input_shape or x_train[0].shape

        # adds a validation set
        if x_valid is not None:
            self.valid_size = self._parse_attr_size(valid_size, x_valid)

        # creates a validation set using a portion of the training
        elif valid_split is not None:
            self.valid_size = int(self.train_size * valid_split)
            self.train_size -= self.valid_size
            self.x_valid, self.y_valid, self.x_train, self.y_train = self.split(
                self.x_train, self.y_train, self.valid_size)

        # adds the number of classes if needed reshapes labels to be categorical
        if num_classes:
            self.num_classes = num_classes
            self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
            self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)
            if self.y_valid is not None:
                self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)

    @classmethod
    def from_pickle(cls, pickle_file: str, **kwargs):
        with open(pickle_file, 'rb') as file:
            data_dict = pickle.load(file)
        return cls(**data_dict, **kwargs)

    @classmethod
    def from_npy(cls):
        pass

    @classmethod
    def from_folder(cls):
        '''
        dataset/
        --image/
        ----image1
        ----image2
        --label/
        ----label1
        ----label2
        '''
        # TODO missing implementation

    def _parse_attr_size(self, value, attr_data):

        return min(len(attr_data), abs(value)) if value else len(attr_data)

    def split(self, data, labels, split_size=None):
        '''Splits the array into two arrays of data.

        # Parameters
        data: list of data
        labels: list of labels
        split_size: integer value related to the split size

        # Return
        two sets of data grouped as (data_a, labels_a), (data_b, labelb_b)\n
        data_a and labels_a have len() of split_size\n
        data_b and labels_b have the remainder of data\n'''

        data_a = data[:split_size]
        data_b = data[split_size:]
        label_a = labels[:split_size]
        label_b = labels[split_size:]
        return data_a, label_a, data_b, label_b

    def shuffle(self, data, labels):
        '''Shuffles a pair of data and labels'''

        indexes = np.random.permutation(np.arange(len(data)))
        return data[indexes], labels[indexes]

    def get_data(self, attr_name: str, sample_size: int=None, shuffle: bool=False):

        if attr_name not in ['train', 'test', 'valid']:
            raise ValueError(f'Unknown set name: {attr_name}')

        x_data = getattr(self, f'x_{attr_name}')
        y_data = getattr(self, f'y_{attr_name}')
        data_size = getattr(self, f'{attr_name}_size')

        if sample_size is not None:
            data_size = sample_size

        if shuffle:
            x_data, y_data = self.shuffle(x_data, y_data)

        return x_data[:data_size], y_data[:data_size]
