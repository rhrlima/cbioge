# import sys; sys.path.append('..')

import json
import numpy as np
import pickle
import re

from math import sin, cos, exp, log

from keras.models import model_from_json
from keras.utils import np_utils


DEBUG = False


class BaseProblem:

    def map_genotype_to_phenotype(self, solution):
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, phenotype):
        raise NotImplementedError('Not implemented yet.')


class CnnProblem(BaseProblem):

    parser = None

    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    x_test = None
    y_test = None

    input_shape = None
    num_classes = None

    batch_size = 128
    epochs = 1

    loss = 'categorical_crossentropy'
    opt = 'adam'
    metrics = ['accuracy']

    def __init__(self, parser_, dataset=None):
        self.parser = parser_
        if dataset:
            self.load_dataset_from_pickle(dataset)

    def load_dataset_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            temp = pickle.load(f)

            self.x_train = temp['train_dataset']
            self.y_train = temp['train_labels']

            self.x_valid = temp['valid_dataset']
            self.y_valid = temp['valid_labels']

            self.x_test = temp['test_dataset']
            self.y_test = temp['test_labels']

            self.input_shape = temp['input_shape']
            self.num_classes = temp['num_classes']

            del temp

        self.x_train = self.x_train.reshape((-1,)+self.input_shape)
        self.x_valid = self.x_valid.reshape((-1,)+self.input_shape)
        self.x_test = self.x_test.reshape((-1,)+self.input_shape)

        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def map_genotype_to_phenotype(self, genotype):
        add_input_shape = True
        add_flatten = True
        add_output_shape = True

        deriv = self.parser.parse(genotype)

        if not deriv:
            return None

        nodes = []
        node = {'class_name': None, 'config': {}}

        index = 0
        while index < len(deriv):

            key, value = deriv[index:index+2]

            if key == 'class_name':

                if node[key] is not None:
                    nodes.append(node)
                    node = {'class_name': None, 'config': {}}

                # first Conv node needs input_shape parameter
                if add_input_shape:
                    node['config']['input_shape'] = self.input_shape
                    add_input_shape = False

                # first Dense node needs Flatten before
                if value == 'Dense' and add_flatten:
                    nodes.append({'class_name': 'Flatten', 'config': {}})
                    add_flatten = False

                node[key] = value
            else:
                # range pattern
                m = re.match('\\[(\\d+[.\\d+]*),\\s*(\\d+[.\\d+]*)\\]', value)
                if m:
                    min_ = eval(m.group(1))
                    max_ = eval(m.group(2))
                    if type(min_) == int and type(max_) == int:
                        value = np.random.randint(min_, max_)
                    elif type(min_) == float and type(max_) == float:
                        value = np.random.uniform(min_, max_)
                    else:
                        raise TypeError('type mismatch')
                else:
                    # kernel size pattern
                    m1 = re.match('\\((\\d+),\\s*(\\d+)\\)', value)
                    m2 = re.match('^\\d+$', value)
                    if m1 or m2:
                        value = eval(value)

                node['config'][key] = value

            index += 2
        else:
            # last node needs output_shape as number of classes
            # and softmax activation
            if add_output_shape:
                node['config']['units'] = self.num_classes
                node['config']['activation'] = 'softmax'
                add_output_shape = False
            nodes.append(node)

        model = {'class_name': 'Sequential', 'config': []}
        for n in nodes:
            if DEBUG:
                print(n)
            model['config'].append(n)

        # returns the model as string
        return json.dumps(model)

    def evaluate(self, solution, verbose=0):

        try:
            json_model = self.map_genotype_to_phenotype(solution.genotype)
            model = model_from_json(json_model)

            if not model:
                return -1, None

            model.compile(
                loss=self.loss,
                optimizer=self.opt,
                metrics=self.metrics)

            # train
            if verbose:
                print('[training]')
            model.fit(self.x_train, self.y_train, batch_size=self.batch_size,
                      epochs=self.epochs, verbose=verbose)

            # valid
            if verbose:
                print('[validation]')
            score = model.evaluate(self.x_valid, self.y_valid, verbose=verbose)

            if verbose:
                print('loss: {}\taccuracy: {}'.format(score[0], score[1]))

            return score[1], json_model

        except Exception as e:
            if DEBUG:
                print(e)
                print('[evaluation] invalid model from solution: {}'.format(
                    solution.genotype))
            return -1, None


class DNNProblem(BaseProblem):

    parser = None

    x_train = None
    y_train = None
    x_valid = None
    y_valid = None
    x_test = None
    y_test = None

    input_shape = None
    num_classes = None

    batch_size = 128
    epochs = 1

    loss = 'categorical_crossentropy'
    opt = 'adam'
    metrics = ['accuracy']

    def __init__(self, parser_, dataset=None):
        self.parser = parser_
        if dataset:
            self.load_dataset_from_pickle(dataset)

    def load_dataset_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            temp = pickle.load(f)

            self.x_train = temp['train_dataset']
            self.y_train = temp['train_labels']

            self.x_valid = temp['valid_dataset']
            self.y_valid = temp['valid_labels']

            self.x_test = temp['test_dataset']
            self.y_test = temp['test_labels']

            self.input_shape = temp['input_shape']
            self.num_classes = temp['num_classes']

            del temp

        self.x_train = self.x_train.reshape((-1,)+self.input_shape)
        self.x_valid = self.x_valid.reshape((-1,)+self.input_shape)
        self.x_test = self.x_test.reshape((-1,)+self.input_shape)

        self.y_train = np_utils.to_categorical(self.y_train, self.num_classes)
        self.y_valid = np_utils.to_categorical(self.y_valid, self.num_classes)
        self.y_test = np_utils.to_categorical(self.y_test, self.num_classes)

    def map_genotype_to_phenotype(self, genotype):
        pass


class SymbolicRegressionProblem(BaseProblem):

    def __init__(self, parser_, eq=None):
        self.parser = parser_
        self.equation = eq
        self.inputs = np.arange(-1, 1, 0.1)
        self.known_best = None

    def map_genotype_to_phenotype(self, genotype):
        deriv = self.parser.parse(genotype)
        if not deriv:
            return None
        return ''.join(deriv)

    def evaluate(self, solution):

        solution.phenotype = self.map_genotype_to_phenotype(solution.genotype)

        if not solution.phenotype:
            return float('inf'), None

        try:
            eq1 = lambda x: eval(solution.phenotype)
            diff_func = lambda x: (eq1(x) - self.equation(x))**2
            diff = sum(map(diff_func, self.inputs))
        except Exception:
            return float('inf'), None

        return diff, solution.phenotype


def inv(x):
    return 1 / x


class StringMatchProblem(BaseProblem):

    def __init__(self, parser_, target='Hello World!'):
        self.parser = parser_
        self.target = target

    def map_genotype_to_phenotype(self, genotype):
        deriv = self.parser.parse(genotype)
        if not deriv:
            return None
        return ''.join(deriv)

    def evaluate(self, solution):

        phen = self.map_genotype_to_phenotype(solution.genotype)

        if not phen:
            return float('inf'), None

        fit = sum([(a == b) for a, b in zip(phen, self.target)]) / len(self.target)

        solution.phenotype = phen
        solution.fitness = fit

        return fit, phen
