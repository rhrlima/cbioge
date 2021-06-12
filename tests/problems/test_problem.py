import pytest

from cbioge.grammars import Grammar
from cbioge.problems import BaseProblem, DNNProblem

def get_mockup_parser():
    return Grammar('tests/grammars/test_grammar.json')

def get_mockup_data_dict():

    return {
        'x_train': [], 
        'y_train': [], 
        'x_valid': [], 
        'y_valid': [], 
        'x_test': [], 
        'y_test': [], 
        'input_shape': (None,)
    }

def test_isinstance_of_baseproblem():
    problem = BaseProblem()
    assert isinstance(problem, BaseProblem)

def test_isinstance_of_dnnproblem():
    parser = get_mockup_parser()
    dataset = get_mockup_data_dict()
    problem = DNNProblem(parser, dataset)
    assert isinstance(problem, DNNProblem)

def test_is_dnnproblem_subclass_of_problem():
    assert issubclass(DNNProblem, BaseProblem) == True

def test_baseproblem_has_functions():
    problem = BaseProblem()
    assert hasattr(problem, 'map_genotype_to_phenotype') == True
    assert hasattr(problem, 'evaluate') == True

def test_dnnproblem_read_valid_dataset():
    parser = get_mockup_parser()
    dataset = get_mockup_data_dict()
    problem = DNNProblem(parser, dataset)
    assert hasattr(problem, 'x_train') == True
    assert hasattr(problem, 'y_train') == True
    assert hasattr(problem, 'x_valid') == True
    assert hasattr(problem, 'y_valid') == True
    assert hasattr(problem, 'x_test') == True
    assert hasattr(problem, 'y_test') == True

    assert hasattr(problem, 'input_shape') == True

    assert hasattr(problem, 'train_size') == True
    assert hasattr(problem, 'valid_size') == True
    assert hasattr(problem, 'test_size') == True

def test_dnnproblem_none_parser():
    with pytest.raises(AttributeError):
        DNNProblem(None, get_mockup_data_dict())

def test_dnnproblem_none_dataset():
    with pytest.raises(AttributeError):
        DNNProblem(get_mockup_parser(), None)

def test_reshape_mapping():
    mapping = ['conv', 32, 'same', 'conv', 32, 'same', 'dense', 10]
    problem = DNNProblem(get_mockup_parser(), get_mockup_data_dict())
    assert problem._reshape_mapping(mapping) == [
        ['conv', 32, 'same'], 
        ['conv', 32, 'same'], 
        ['dense', 10]
    ]

def test_building_json_model():
    mapping = ['conv', 32, 'same', 'conv', 32, 'same', 'dense', 10]
    problem = DNNProblem(get_mockup_parser(), get_mockup_data_dict())
    mapping = problem._reshape_mapping(mapping)
    model = problem._base_build(mapping)
    problem._wrap_up_model(model)
    assert model == {
        'class_name': 'Model', 'config': {'layers': [
            {'class_name': 'Conv2D', 'config': {
                'filters': 32, 'padding': 'same'}, 'inbound_nodes': [], 'name': 'conv_0'},
            {'class_name': 'Conv2D', 'config': {
                'filters': 32, 'padding': 'same'}, 'inbound_nodes': [[['conv_0', 0, 0]]], 'name': 'conv_1'},
            {'class_name': 'Dense', 'config': {'units': 10}, 
                'inbound_nodes': [[['conv_1', 0, 0]]], 'name': 'dense_0'}
        ], 'input_layers': [['conv_0', 0, 0]], 'output_layers': [['dense_0', 0, 0]]}}
