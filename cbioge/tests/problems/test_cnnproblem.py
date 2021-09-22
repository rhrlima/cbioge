import os
import pytest

from cbioge.algorithms import Solution
from cbioge.grammars import Grammar
from cbioge.datasets import Dataset
from cbioge.problems import CNNProblem

def get_mockup_data_dict():

    return Dataset({
        'x_train': [], 
        'y_train': [], 
        'x_test': [], 
        'y_test': [], 
        'input_shape': (32, 32, 1), 
        'num_classes': 1
    })

def get_mockup_parser():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(base_dir, 'assets', 'test_grammar.json')

def test_map_genotype_to_phenotype():
    parser = Grammar(get_mockup_parser())
    solution = Solution([[5], [0], [0, 0], [2], [0], [0, 1]])
    mapping = [['conv', 16, 4], ['dense', 32], ['dense', 64]]

    problem = CNNProblem(parser, get_mockup_data_dict())
    model = problem.map_genotype_to_phenotype(solution)
    model2 = problem._sequential_build([['conv', 16, 4], ['dense', 32], ['dense', 64]])

    assert solution.data['mapping'] == mapping
    assert model.count_params() == model2.count_params()