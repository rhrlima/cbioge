import numpy as np
import pickle

import keras
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

EPOCHS = 5
BATCH_SIZE = 128
VERBOSE = True

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

model.add(Conv2D(filters=32, 
				 kernel_size=(3, 3), 
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

opt = keras.optimizers.Adam(
	lr=0.01, 
	beta_1=0.9, 
	beta_2=0.999, 
	epsilon=1.0 * 10**-8, 
	decay=0.001, 
	amsgrad=False)

model.compile(
	loss='categorical_crossentropy', 
	optimizer=opt, 
	metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(
	monitor='val_loss', 
	min_delta=0, 
	patience=2, 
	verbose=0, 
	mode='auto')

hist = model.fit(x_train, y_train, 
	batch_size=BATCH_SIZE, 
	epochs=EPOCHS, 
	verbose=VERBOSE, 
	validation_data=(x_valid, y_valid), 
	callbacks=[es]
)

print('train\nloss: {}\taccuracy: {}'.format(
	np.mean(hist.history['loss']), 
	np.mean(hist.history['acc']))
)

score = model.evaluate(x_test, y_test, verbose=VERBOSE)
print('test\nloss: {}\taccuracy: {}'.format(score[0], score[1]))