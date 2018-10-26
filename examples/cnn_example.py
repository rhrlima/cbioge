import numpy as np
import pickle

from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential, model_from_json
#from keras.callbacks import History
from keras.utils import np_utils

#file = '../datasets/mnist/mnist.pickle'
#file = '../datasets/notmnist/notMNIST.pickle'
#file = '../datasets/fashion-mnist/fashion-mnist.pickle'
#file = '../datasets/cifar-10/cifar-10.pickle'
file = '../datasets/cifar-100/cifar-100.pickle'

with open(file, 'rb') as f:
	data = pickle.load(f)
x_train = data['train_dataset']
y_train = data['train_labels']
x_valid = data['valid_dataset']
y_valid = data['valid_labels']
x_test  = data['test_dataset']
y_test  = data['test_labels']
input_shape = data['input_shape']
num_classes = data['num_classes']
del data

x_train = x_train.reshape((-1,)+input_shape)
x_valid = x_valid.reshape((-1,)+input_shape)
x_test = x_test.reshape((-1,)+input_shape)

y_train = np_utils.to_categorical(y_train, num_classes)
y_valid = np_utils.to_categorical(y_valid, num_classes)
y_test  = np_utils.to_categorical(y_test , num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), 
	activation='relu', 
	input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#hist = History()
hist = model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)

print('train', hist.history['acc'])

score = model.evaluate(x_valid, y_valid, verbose=1)
print('validation\nloss: {}\taccuracy: {}'.format(score[0], score[1]))

score = model.evaluate(x_test, y_test, verbose=1)
print('test\nloss: {}\taccuracy: {}'.format(score[0], score[1]))