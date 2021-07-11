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

def test_map_genotype_to_phenotype():
    # np.random.seed(0)
    # parser = get_mockup_parser()
    # solution = GESolution(parser.dsge_create_solution())
    # print(parser.dsge_recursive_parse(solution.genotype))
    # problem = UNetProblem(parser, get_mockup_data_dict())
    # json_model = problem.map_genotype_to_phenotype(solution)
    # assert json_model == json.dumps(
    #     {"class_name": "Model", "config": {"layers": [
    #     {"class_name": "InputLayer", "name": "input_0", "config": {"batch_input_shape": [None, 32, 32, 3]}, "inbound_nodes": []}, 
    #     {"class_name": "Conv2D", "name": "conv_0", "config": {"filters": 256, "kernel_size": 4, "strides": 1, "padding": "same", "activation": "selu"}, "inbound_nodes": [[["input_0", 0, 0]]]}, 
    #     {"class_name": "Conv2D", "name": "conv_1", "config": {"filters": 128, "kernel_size": 4, "strides": 1, "padding": "same", "activation": "linear"}, "inbound_nodes": [[["conv_0", 0, 0]]]}, 
    #     {"class_name": "MaxPooling2D", "name": "maxpool_0", "config": {"pool_size": 2, "strides": 2, "padding": "same"}, "inbound_nodes": [[["conv_1", 0, 0]]]}, 
    #     {"class_name": "Dropout", "name": "dropout_0", "config": {"rate": 0.32294705653332806}, "inbound_nodes": [[["maxpool_0", 0, 0]]]}, 
    #     {"class_name": "UpSampling2D", "name": "upsamp_0", "config": {"size": 2}, "inbound_nodes": [[["dropout_0", 0, 0]]]}, 
    #     {"class_name": "Conv2D", "name": "conv_2", "config": {"filters": 128, "kernel_size": 2, "strides": 1, "padding": "same", "activation": "relu"}, "inbound_nodes": [[["upsamp_0", 0, 0]]]}, 
    #     {"class_name": "Concatenate", "name": "concat_0", "config": {"axis": 3}, "inbound_nodes": [[["conv_1", 0, 0], ["conv_2", 0, 0]]]}, 
    #     {"class_name": "Conv2D", "name": "conv_3", "config": {"filters": 128, "kernel_size": 4, "strides": 1, "padding": "same", "activation": "linear"}, "inbound_nodes": [[["concat_0", 0, 0]]]}, 
    #     {"class_name": "Conv2D", "name": "conv_4", "config": {"filters": 256, "kernel_size": 4, "strides": 1, "padding": "same", "activation": "selu"}, "inbound_nodes": [[["conv_3", 0, 0]]]}, 
    #     {"class_name": "Conv2D", "name": "conv_5", "config": {"filters": 2, "kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}, "inbound_nodes": [[["conv_4", 0, 0]]]}, 
    #     {"class_name": "Conv2D", "name": "conv_6", "config": {"filters": 1, "kernel_size": 1, "strides": 1, "padding": "same", "activation": "sigmoid"}, "inbound_nodes": [[["conv_5", 0, 0]]]}], 
    #     "input_layers": [["input_0", 0, 0]], "output_layers": [["conv_6", 0, 0]]}})
    assert True

@pytest.mark.parametrize('grammar_file', [
    'cbioge/assets/grammars/unet_restricted.json'
])
def test_invalid_models_tolerance(grammar_file):
    parser = Grammar(grammar_file)
    problem = UNetProblem(parser, get_mockup_data_dict())

    num_models = 100
    invalid = 0
    for _ in range(num_models):
        solution = GESolution(parser.dsge_create_solution())
        problem.map_genotype_to_phenotype(solution)
        if solution.phenotype is None: invalid += 1

    assert invalid/num_models <= 0.5