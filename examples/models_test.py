import pickle
import json

from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.utils import np_utils

from keras.models import model_from_json


def functional_model():
    inputs = Input(shape=(784,))

    layer = Dense(64, activation='relu')(inputs)
    layer = Dense(64, activation='relu')(layer)
    predictions = Dense(10, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=predictions)
    return model


def functional_model_from_json():
    json_model = {
        "class_name": "Model", 
        "config": {
            "name": "model_1", 
            "layers": [
                {
                    "name": "input_1", 
                    "class_name": "InputLayer", 
                    "config": {
                        "batch_input_shape": [None, 784], 
                        "dtype": "float32", 
                        "name": "input_1"
                    }, "inbound_nodes": []
                }, 
                {
                    "name": "dense_4", 
                    "class_name": "Dense", 
                    "config": {
                        "name": "dense_4", 
                        "units": 64, 
                        "activation": "relu", 
                    }, 
                    "inbound_nodes": [
                        [
                            ["input_1", 0, 0, {}]
                        ]
                    ]
                }, 
                {
                    "name": "dense_5", 
                    "class_name": "Dense", 
                    "config": {
                        "name": "dense_5", 
                        "units": 64, 
                        "activation": "relu", 
                    }, "inbound_nodes": [[["dense_4", 0, 0, {}]]]
                }, 
                {
                    "name": "dense_6", 
                    "class_name": "Dense", 
                    "config": {
                        "name": "dense_6", 
                        "units": 10, 
                        "activation": "softmax", 
                    }, 
                    "inbound_nodes": [[["dense_5", 0, 0, {}]]]
                }
            ], 
            "input_layers": [["input_1", 0, 0]], 
            "output_layers": [["dense_6", 0, 0]]
        }, 
        "keras_version": "2.2.2", 
        "backend": "tensorflow"
    }
    return model_from_json(json.dumps(json_model))


def sequential_model():
    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model


def run_model(model, x_train, y_train, x_valid, y_valid):
    print(model.to_json())
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=1)
    score = model.evaluate(x_valid, y_valid)
    print('test\nloss: {}\taccuracy: {}'.format(score[0], score[1]))


if __name__ == '__main__':

    file = '../datasets/mnist.pickle'

    EPOCHS = 5
    BATCH_SIZE = 128
    VERBOSE = True

    with open(file, 'rb') as f:
        data = pickle.load(f)
    x_train = data['train_dataset']
    y_train = data['train_labels']
    x_valid = data['valid_dataset']
    y_valid = data['valid_labels']
    x_test = data['test_dataset']
    y_test = data['test_labels']
    input_shape = (784,)
    num_classes = data['num_classes']
    del data

    x_train = x_train.reshape((-1,)+input_shape)
    x_valid = x_valid.reshape((-1,)+input_shape)
    x_test = x_test.reshape((-1,)+input_shape)

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_valid = np_utils.to_categorical(y_valid, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    #print('Sequential model')
    #model = sequential_model()
    #run_model(model, x_train, y_train, x_valid, y_valid)

    #print('Functional model')
    #model = functional_model()
    #run_model(model, x_train, y_train, x_valid, y_valid)

    print('Functional model from json')
    model = functional_model_from_json()
    run_model(model, x_train, y_train, x_valid, y_valid)
