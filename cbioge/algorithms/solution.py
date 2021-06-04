import copy

import numpy as np

class GESolution():

    def __init__(self, gen=[], json_data=None):

        self.id = None
        self.genotype = gen
        self.phenotype = None
        self.fitness = -1
        self.evaluated = False
        self.time = None
        self.params = None

        if json_data is not None:
            self.initialize_from_json(json_data)

    def __str__(self):
        return str(self.genotype)

    def to_json(self):
        return self.__dict__

    def initialize_from_json(self, json_data):
        for key in self.__dict__:
            self.__dict__[key] = json_data[key]

    def copy(self):
        new_solution = copy.deepcopy(self)
        new_solution.evaluated = False
        return new_solution
