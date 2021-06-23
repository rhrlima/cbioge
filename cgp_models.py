from cbioge.algorithms.solution import GESolution
import os

from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem
from cbioge.datasets.dataset import read_dataset_from_pickle
from cbioge.problems.dnn import ModelRunner

from cbioge.utils.experiments import check_os

from keras.datasets import cifar10
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import np_utils

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

def CGP_best_resset(input_shape, num_classes):
    
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

def model_from_mapping():

    mapping = [
        ['resblock', 128, 3], 
        ['resblock', 128, 3], 
        ['resblock', 128, 3], 

        ['avgpool', 2, 2, 'valid'], 

        ['resblock', 32, 3], 
        ['resblock', 128, 3], 
        ['resblock', 128, 3], 

        ['avgpool', 2, 2, 'valid'], 
        ['avgpool', 2, 2, 'valid'], 

        ['resblock', 128, 3], 

        ['maxpool', 2, 2, 'valid'], 

        ['dense', 256, 'relu'],
    ]

    return problem.sequential_build(mapping)


if __name__ == '__main__':

    import logging
    logging.getLogger('tensorflow').disabled = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    check_os()
  
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train / 255
    # x_test = x_test / 255
    # y_train = np_utils.to_categorical(y_train, 10)
    # y_test = np_utils.to_categorical(y_test, 10)
    # train_size = 5000

    np.random.seed(0)
    parser = Grammar('data/grammars/text_test.json')
    problem = CNNProblem(
        parser, 
        read_dataset_from_pickle('data/datasets/cifar10.pickle'), 
        batch_size=128, 
        epochs=50)

    genotype = parser.dsge_create_solution()
    mapping, genotype = parser.dsge_recursive_parse(genotype)
    print('# result')
    for m in mapping:
        print(m)

    # model = model_from_mapping()
    # #model = CGP_ResSet((32,32,3), 10)
    # #model = CGP_best_resset((32,32,3), 10)
    # model.compile(
    #     loss=problem.loss, 
    #     optimizer='SGD',
    #     metrics=['accuracy']
    # )

    # model.summary()

    # wpath='cgp_test'
    # runner = ModelRunner(model, path=wpath)
    # runner.train_model(x_train[:train_size], y_train[:train_size], 
    #     #validation_data=(problem.x_valid, problem.y_valid), 
    #     batch_size=problem.batch_size, 
    #     epochs=problem.epochs, 
    #     timelimit=problem.timelimit, 
    #     save_weights=True, 
    #     shuffle=True, 
    #     verbose=True)

    # runner.test_model(x_test, y_test, 
    #     batch_size=problem.batch_size, 
    #     weights_path=os.path.join(wpath, 'weights.hdf5'))

    # print(runner.loss, runner.accuracy)