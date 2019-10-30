import keras
from keras.models import *
from keras.layers import *

def common_layers(y):
	y = BatchNormalization()(y)
	y = LeakyReLU()(y)
	return y

input_shape = (64, 64, 1)

inputs = Input(input_shape)

#preres = Conv2D(64, (3, 3), activation='relu')(inputs)

layer = Conv2D(64, 1)(inputs)
layer = common_layers(layer)
layer = Conv2D(64, 3, padding='same')(layer)
layer = common_layers(layer)
layer = Conv2D(256, 1)(layer)
layer = common_layers(layer)

layer = BatchNormalization()(layer)

# upscale when size dont match
aux = Conv2D(256, 1, strides=1)(inputs)
aux = common_layers(aux)
#aux = BatchNormalization()(aux)

output = Add()([layer, aux])

model = Model(input=inputs, output=output)

model.summary()

print(model.to_json())