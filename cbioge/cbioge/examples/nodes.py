from keras.layers import *
from keras.models import *


def linear():
	layers = []

	layers.append(Input((32, 32, 1)))
	layers.append(Conv2D(1, 1)(layers[-1]))
	layers.append(Conv2D(1, 1)(layers[-1]))
	layers.append(Dense(1)(layers[-1]))

	model = Model(inputs=layers[0], outputs=layers[-1])
	model.summary()


def skip():
	layers = []

	layers.append(Input((32, 32, 1)))
	layers.append(Conv2D(1, 1)(layers[-1]))
	layers.append(Conv2D(1, 1)(layers[-1]))
	layers.append(concatenate([layers[0], layers[2]], axis=3))
	layers.append(Dense(1)(layers[-1]))

	model = Model(inputs=layers[0], outputs=layers[-1])
	model.summary()


def split():
	layers = []

	layers.append(Input((32, 32, 1)))
	layers.append(Conv2D(1, 1)(layers[0]))
	layers.append(Conv2D(1, 1)(layers[0]))
	layers.append(Conv2D(1, 1)(layers[0]))

	model = Model(inputs=layers[0], outputs=layers[1:])
	model.summary()


def join():
	layers = []

	layers.append(Input((32, 32, 1)))
	layers.append(Input((32, 32, 1)))
	layers.append(Input((32, 32, 1)))
	layers.append(concatenate(layers[:], axis=3))

	model = Model(inputs=layers[:3], outputs=layers[-1])
	model.summary()


if __name__ == '__main__':
	
	linear()
	skip()
	split()
	join()