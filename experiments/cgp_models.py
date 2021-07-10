from keras.models import *
from keras.layers import *


def res_block(inputs, filters, kernel_size):
    temp = Conv2D(filters, 1, padding='same')(inputs)
    output = Conv2D(filters, kernel_size, padding='same')(inputs)
    output = Conv2D(filters, kernel_size, padding='same')(output)
    output = BatchNormalization()(output)
    output = Add()([output, temp])
    output = ReLU()(output)
    return output


def CGP_ResSet(input_shape, num_classes):

    inputs = Input(input_shape)

    layer = res_block(inputs, 128, 3)
    layer = res_block(layer, 128, 3)
    layer = res_block(layer, 128, 3)

    layer = AveragePooling2D()(layer)

    layer = res_block(layer, 32, 3)
    layer = res_block(layer, 128, 3)
    layer = res_block(layer, 128, 3)

    layer = AveragePooling2D()(layer)
    layer = AveragePooling2D()(layer)

    layer = res_block(layer, 128, 3)

    layer = MaxPooling2D()(layer)

    layer = Flatten()(layer)
    layer = Dense(num_classes, activation='softmax')(layer)
    
    return Model(inputs=inputs, outputs=layer)


def CGP_best_ResSet(input_shape, num_classes):
    
    inputs = Input(input_shape)

    rb1 = res_block(inputs, 64, 3)
    rb2 = res_block(inputs, 64, 5)
    layer = Add()([rb1, rb2])

    layer = res_block(layer, 128, 5)
    layer = AveragePooling2D()(layer) 
    layer = res_block(layer, 128, 5)
    layer = MaxPooling2D()(layer)
    layer = res_block(layer, 128, 5)
    layer = res_block(layer, 32, 5)
    layer = MaxPooling2D()(layer)
    layer = res_block(layer, 64, 5)
    layer = AveragePooling2D()(layer)

    layer = Flatten()(layer)
    layer = Dense(num_classes, activation='softmax')(layer)
    
    return Model(inputs=inputs, outputs=layer)

def test(input_shape, num_classes):

    inputs = Input(input_shape)
    output = Conv2D(32, 3)(inputs)
    output = Conv2D(16, 3)(output)
    output = Dense(64)(output)
    output = Flatten()(output)
    output = Dense(num_classes, activation='softmax')(output)
    model = Model(inputs=inputs, outputs=output)
    return model

