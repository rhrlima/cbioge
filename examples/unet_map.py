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

	np.random.seed(0)

	dset = {
		'input_shape': (256, 256, 1)
	}

	parser = BNFGrammar('grammars/unet_mirror.bnf')
	problem = UNetProblem(parser, dset)

	num = 1
	failed = 0
	for i in range(num):
		gen = parser.dsge_create_solution()
		print(gen)
		fen = parser.dsge_recursive_parse(gen)
		print(problem._reshape_mapping(fen))
		model = problem._map_genotype_to_phenotype(gen)
		problem._repair_genotype(gen, model)
		#try:
		#model = model_from_json(model)
		#except:
		#	failed += 1
		print(f'\r\r{failed}/{i+1} {failed/(i+1)}%', end='')

	#print('failed', failed, failed/num)
	# if model:
	# 	model.summary()
	# 	model.compile('adam', loss='binary_crossentropy')
	# 	print('ALL GOOD')

	#unet(dset['input_shape'])