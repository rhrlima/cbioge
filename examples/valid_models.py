from keras.layers import *
from keras.models import *

import itertools


if __name__ == '__main__':

	kernels = [1, 2, 3, 4]
	strides = [1, 2]
	padding = ['same', 'valid']

	configs = list(itertools.product(kernels, strides, padding))

	# for cfg in configs:
	# 	print(cfg)
	# 	inp = Input((8, 8,1))
	# 	lay = Conv2D(32, cfg[0], strides=cfg[1], padding=cfg[2])(inp)
	# 	lay = Conv2DTranspose(32, cfg[0], strides=cfg[1], padding=cfg[2])(lay)
	# 	model = Model(inputs=inp, outputs=lay)
	# 	model.summary()

	for size in range(1, 10):
		try:
			print(size)
			inp = Input((size, size, 1))
			lay = Conv2D(32, 3, strides=2, padding='valid')(inp)
			lay = Conv2DTranspose(32, 3, strides=2, padding='valid')(lay)
			lay = Conv2DTranspose(32, 2, strides=1, padding='valid')(lay)
			model = Model(inputs=inp, outputs=lay)
			model.summary()
		except:
			continue