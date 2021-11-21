import os
import numpy as np

import pytest, pickle

from cbioge.datasets import Dataset

def get_mockup_dataset(keys: list=None):
    data = {
        'x_train': np.zeros((100, 10, 10)),
        'y_train': np.zeros((100, 1)),

        'x_valid': np.zeros((100, 10, 10)),
        'y_valid': np.zeros((100, 1)),

        'x_test': np.zeros((100, 10, 10)),
        'y_test': np.zeros((100, 1)),

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
    dataset = Dataset(**get_mockup_dataset())
    assert dataset is not None

def test_load_dataset_from_pickle():
    pickle_file = get_mockup_pickle_file()
    dataset = Dataset.from_pickle(pickle_file)
    assert dataset is not None

@pytest.mark.parametrize('field', ['train', 'test', 'valid'])
@pytest.mark.parametrize('value', [None, 0, 10, 50, 100, -10, 200, -200])
def test_attr_sizes(field, value):

    data_dict = get_mockup_dataset()

    attr_data = f'x_{field}'
    attr_size = f'{field}_size'

    data_dict[attr_size] = value

    dataset = Dataset(**data_dict)

    if value: expected = min(abs(value), len(data_dict[attr_data]))
    else: expected = len(data_dict[attr_data])

    assert getattr(dataset, attr_size) == expected
    assert getattr(dataset, attr_size) >= 0
    assert len(getattr(dataset, attr_data)) >= getattr(dataset, attr_size)

@pytest.mark.parametrize('value', [None, 0.0, 0.1, 0.5, 1.0, -0.1, 2.0, -2.0])
def test_valid_split(value):

    test_attr_sizes('valid', value * 100 if value else None)

@pytest.mark.parametrize('data_name', ['train', 'test', 'valid'])
def test_valid_get_data(data_name):

    data_dict = get_mockup_dataset()

    dataset = Dataset(**data_dict)

    x_data_name = f'x_{data_name}'
    y_data_name = f'y_{data_name}'

    x_data, y_data = dataset.get_data(data_name)
    x_mock, y_mock = data_dict[x_data_name], data_dict[y_data_name]

    assert x_data.all() == x_mock.all()
    assert y_data.all() == y_mock.all()
    assert len(x_data) == len(x_mock)

@pytest.mark.parametrize('data_name', ['other', None])
def test_invalid_get_data(data_name):

    with pytest.raises(ValueError):
        Dataset(**get_mockup_dataset()).get_data(data_name)

def test_split():

    dataset = Dataset(**get_mockup_dataset())

    split_size = 50
    dA, lA, dB, lB = dataset.split(dataset.x_train, dataset.y_train, split_size)

    assert len(dA) == split_size
    assert len(lA) == split_size
    assert len(dB) == len(dataset.x_train) - split_size
    assert len(lB) == len(dataset.y_train) - split_size

def test_shuffle():
    pass
