import numpy as np

from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.utils import np_utils

from matplotlib import pyplot as plt

K.set_image_dim_ordering('th')
np.random.seed(123)

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

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', data_format='channels_first'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=1, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print('score', score)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
	json_file.write(model_json)
model.save_weights('model.h5')
print('Model saved to disk')

print('------')

with open('model.json', 'r') as json_file:
	model_json = json_file.read()
model2 = model_from_json(model_json)
model2.load_weights('model.h5')
print('Loaded model from disk')

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(X_test, Y_test, verbose=0)
print('score2', score)