import pytest

from cbioge.grammars import Grammar
from cbioge.datasets import Dataset
from cbioge.problems import UNetProblem
from cbioge.algorithms import GESolution

def get_mockup_data_dict():

    return Dataset({
        'x_train': [], 
        'y_train': [], 
        'x_test': [], 
        'y_test': [], 
        'input_shape': (32, 32, 1), 
        'num_classes': 1
    })