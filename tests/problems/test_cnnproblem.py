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
        'input_shape': (None,), 
        'num_classes': 1
    }

def test_map_genotype_to_phenotype():
    np.random.seed(0)
    parser = get_mockup_parser()
    genotype = parser.dsge_create_solution()
    problem = CNNProblem(parser, get_mockup_data_dict())
    json_model = problem.map_genotype_to_phenotype(genotype)
    assert json_model == json.dumps({
        "class_name": "Model", "config": {"layers": [
            {"class_name": "InputLayer", "name": "input_0", 
                "config": {"batch_input_shape": [None, None]}, "inbound_nodes": []}, 
            {"class_name": "Conv2D", "name": "conv_0", 
                "config": {"filters": 256, "kernel_size": 4, "strides": 2, "padding": "valid", "activation": "tanh"}, "inbound_nodes": [[["input_0", 0, 0]]]}, 
            {"class_name": "Flatten", "name": "flatten_0", 
                "config": {}, "inbound_nodes": [[["conv_0", 0, 0]]]}, 
            {"class_name": "Dense", "name": "dense_0", 
                "config": {"units": 64, "activation": "tanh"}, "inbound_nodes": [[["flatten_0", 0, 0]]]}, 
            {"class_name": "Dense", "name": "dense_1", 
                "config": {"units": 1, "activation": "softmax"}, "inbound_nodes": [[["dense_0", 0, 0]]]}], 
        "input_layers": [["input_0", 0, 0]], 
        "output_layers": [["dense_1", 0, 0]]}
    })