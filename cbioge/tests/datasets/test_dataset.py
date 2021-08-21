import os
import numpy as np

import pytest, pickle

from cbioge.datasets import Dataset

def get_mockup_dataset(keys=None):
    data = {
        'x_train': np.zeros((100, 10, 10)),
        'y_train': np.zeros((10, 1)),

        'x_valid': np.zeros((100, 10, 10)),
        'y_valid': np.zeros((10, 1)),

        'x_test': np.zeros((100, 10, 10)),
        'y_test': np.zeros((10, 1)),

        'input_shape': (10, 10),
        'num_classes': 10
    }
    if keys is None:
        return data
    return {k: data[k] for k in keys}

def get_mockup_pickle_file():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    pickle_file = os.path.join(base_dir, 'assets', 'pickle_dataset.pickle')
    if not os.path.exists(pickle_file):
        data = get_mockup_dataset()
        with open(pickle_file, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return pickle_file

def test_load_dataset_from_dict():
    dataset = Dataset(get_mockup_dataset())
    assert dataset is not None

def test_load_dataset_from_pickle():
    pickle_file = get_mockup_pickle_file()
    dataset = Dataset.from_pickle(pickle_file)
    assert dataset is not None

@pytest.mark.parametrize('keys', [
    None, 
    ['input_shape', 'x_train', 'y_train', 'x_test', 'y_test'], 
    ['input_shape', 'x_train', 'y_train', 'x_test', 'y_test', 'num_classes'], 
    ['input_shape', 'x_train', 'y_train', 'x_test', 'y_test', 'x_valid', 'y_valid'], 
    ['input_shape', 'x_train', 'y_train', 'x_test', 'y_test', 'x_valid', 'y_valid', 'num_classes']])
def test_dataset_attributes(keys):
    dataset = Dataset(get_mockup_dataset(keys))
    if keys is None:
        keys = ['input_shape', 'x_train', 'y_train', 'x_test', 'y_test', 'x_valid', 'y_valid', 'num_classes']
    for k in keys:
        assert hasattr(dataset, k)

@pytest.mark.parametrize('train_size', [None, 10, 50])
@pytest.mark.parametrize('test_size', [None, 10, 50])
@pytest.mark.parametrize('valid_size', [None, 10, 50])
@pytest.mark.parametrize('valid_split', [None, 0.1, 0.5])
def test_dataset_sizes(train_size, test_size, valid_size, valid_split):
    
    data_dict = get_mockup_dataset()

    kwargs = {}
    if train_size is not None:
        kwargs['train_size'] = train_size
    else:
        train_size = len(data_dict['x_train'])

    if test_size is not None:
        kwargs['test_size'] = test_size
    else:
        test_size = len(data_dict['x_test'])

    if valid_size is not None:
        kwargs['valid_size'] = valid_size
    else:
        valid_size = len(data_dict['x_valid'])

    if valid_split is not None:
        kwargs['test_split'] = valid_split

    dataset = Dataset(data_dict, **kwargs)

    assert dataset.train_size == train_size
    assert dataset.test_size == test_size
    assert dataset.valid_size == valid_size