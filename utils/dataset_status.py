import pickle
import sys

dataset_file = sys.argv[1]

with open(dataset_file, 'rb') as f:
    data = pickle.load(f)
    x_train = data['train_dataset']
    y_train = data['train_labels']
    x_valid = data['valid_dataset']
    y_valid = data['valid_labels']
    x_test = data['test_dataset']
    y_test = data['test_labels']
    input_shape = data['input_shape']
    num_classes = data['num_classes']
    del data

print('train_dataset', x_train.shape, y_train.shape)
print('valid_dataset', x_valid.shape, y_valid.shape)
print('test_dataset', x_test.shape, y_test.shape)
print('input_shape', input_shape)
print('num_classes', num_classes)
