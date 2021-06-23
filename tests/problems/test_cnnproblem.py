from cbioge.algorithms.solution import GESolution
import json

import numpy as np

from cbioge.grammars import Grammar
from cbioge.problems import CNNProblem

def get_mockup_parser():
    return Grammar('data/grammars/cnn.json')

def get_mockup_data_dict():

    return {
        'x_train': [], 
        'y_train': [], 
        'x_valid': [], 
        'y_valid': [], 
        'x_test': [], 
        'y_test': [], 
        'input_shape': (32, 32, 1), 
        'num_classes': 1
    }

def test_map_genotype_to_phenotype():
    parser = Grammar('tests/data/test_grammar.json')
    solution = GESolution([[5], [0], [0, 0], [2], [0], [0, 1]])
    problem = CNNProblem(parser, get_mockup_data_dict())
    model = problem.map_genotype_to_phenotype(solution)
    model2 = problem.sequential_build([['conv', 16, 4], ['dense', 32], ['dense', 64]])
    assert model.count_params() == model2.count_params()
