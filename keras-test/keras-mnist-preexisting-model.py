import numpy as np

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.utils import np_utils

#K.set_image_dim_ordering('th')
np.random.seed(123)

print('Using pre-existing model to evaluate MNIST')

#--------------------------------------------------------

print('Loading dataset')


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('train shape', X_train.shape, y_train.shape)

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#--------------------------------------------------------

print('Loading model from file')

with open('model.json', 'r') as json_file:
	model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights('model.h5')

layers = model.get_weights()# 8
layer = layers[0]
c = layer[0]

print('layers', len(layers))
print('layer', len(layers[0]))
print(len(c), len(c[0]))
#print('1', len(model.get_weights()[0][0]))
#print('2', len(model.get_weights()[0][0][0]))
#print('3', len(model.get_weights()[0][0][0][0]))

#--------------------------------------------------------

print('Running')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Score -> loss: {}\taccuracy: {}'.format(score[0], score[1]))