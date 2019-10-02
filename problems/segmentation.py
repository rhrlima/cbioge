import os

import keras
from keras.preprocessing.image import ImageDataGenerator

from .problem import BaseProblem
from datasets.dataset import DataGenerator


class ImageSegmentationProblem(BaseProblem):

    parser = None

    batch_size = 1
    epochs = 1

    loss = 'binary_crossentropy'
    opt = 'Adam'
    metrics = ['accuracy']

    def __init__(self, parser, dataset):
        self.parser = parser
        self.dataset = dataset

        self.train_generator = None
        self.test_generator = None
        self.data_augmentation = None

        self.verbose = False

    def map_genotype_to_phenotype(self, genotype):
        return None

    def evaluate(self, solution):

        model = solution.phenotype
        if not model:
            return -1

        model.compile(
            optimizer=self.opt, 
            loss=self.loss, 
            metrics=self.metrics)

        model.fit_generator(
            self.train_generator, 
            self.dataset['train_steps'], 
            self.epochs, verbose=self.verbose)

        loss, acc = model.evaluate_generator(
            self.test_generator, 
            self.dataset['test_steps'], verbose=self.verbose)

        return acc