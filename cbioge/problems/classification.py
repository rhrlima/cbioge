import json
import os
import re
import numpy as np

from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils import np_utils

from cbioge.problems import DNNProblem
from cbioge.problems.dnn import ModelRunner
from cbioge.utils import checkpoint as ckpt

class CNNProblem(DNNProblem):
    ''' Problem class for problems related to classification tasks for DNNs.
        This class includes methods focused on the design of CNNs.
    '''

    def __init__(self, parser, dataset):
        super().__init__(parser, dataset)

        # classification specific
        self.loss = 'categorical_crossentropy'

    def _read_dataset(self, data_dict):
        ''' Reads a dataset stored in a dict. Check parent class for details.

            As a classification specific behavior, this method also reads
            the 'num_classes' key, and reshapes the labels to categorical.
        '''

        super()._read_dataset(data_dict)
        
        self.num_classes = data_dict['num_classes']
        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def map_genotype_to_phenotype(self, genotype):

        mapping, genotype = self.parser.dsge_recursive_parse(genotype)
        mapping = self._reshape_mapping(mapping)

        mapping.insert(0, ['input', (None,)+self.input_shape]) # input layer
        mapping.append(['dense', self.num_classes, 'softmax']) # output layer

        model = self._base_build(mapping)

        self._wrap_up_model(model)

        return json.dumps(model)
