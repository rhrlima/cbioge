import pytest

from cbioge.grammars import Grammar
from cbioge.problems import BaseProblem, DNNProblem

def get_mockup_parser():
    return Grammar('tests/data/test_grammar.json')

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