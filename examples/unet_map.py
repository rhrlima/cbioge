import numpy as np

from keras.models import *
from keras.layers import *

from problems import UNetProblem
from grammars import BNFGrammar

def unet(input_shape):
	in_layer = Input(input_shape)
	layer = Conv2D(32, 3, strides=1, padding='same', activation='relu')(in_layer)
	a = Conv2D(32, 3, strides=1, padding='same', activation='relu')(layer)
	layer = MaxPooling2D(2)(a)
	layer = Conv2D(32, 3, strides=1, padding='same', activation='relu')(layer)
	layer = UpSampling2D(2)(layer)
	layer = Concatenate(axis=3)([a, layer])
	layer = Conv2D(32, 3, strides=1, padding='same', activation='relu')(layer)
	layer = Conv2D(32, 3, strides=1, padding='same', activation='relu')(layer)
	model = Model(inputs=in_layer, outputs=layer)
	model.summary()
	#print(model.to_json())

if __name__ == '__main__':

	#np.random.seed(0)

	dset = {
		'input_shape': (32, 32, 1)
	}

	parser = BNFGrammar('grammars/unet.bnf')
	problem = UNetProblem(parser, dset)

	for _ in range(10):
		gen = parser.dsge_create_solution()
		print(gen)
		fen = parser.dsge_recursive_parse(gen)
		print(fen)
		model = problem._map_genotype_to_phenotype(gen)
		model = model_from_json(model)
	# if model:
	# 	model.summary()
	# 	model.compile('adam', loss='binary_crossentropy')
	# 	print('ALL GOOD')

	#unet(dset['input_shape'])