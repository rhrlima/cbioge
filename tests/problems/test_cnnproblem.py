import pytest

from cbioge.algorithms import GESolution
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

def test_map_genotype_to_phenotype():
    parser = Grammar('cbioge/assets/grammars/test_grammar.json')
    solution = GESolution([[5], [0], [0, 0], [2], [0], [0, 1]])
    mapping = [['conv', 16, 4], ['dense', 32], ['dense', 64]]

    problem = CNNProblem(parser, get_mockup_data_dict())
    model = problem.map_genotype_to_phenotype(solution)
    model2 = problem._sequential_build([['conv', 16, 4], ['dense', 32], ['dense', 64]])

    assert solution.data['mapping'] == mapping
    assert model.count_params() == model2.count_params()

@pytest.mark.parametrize('grammar_file', [
    'cbioge/assets/grammars/test_grammar.json', 
    'cbioge/assets//grammars/cnn3.json',
    'cbioge/assets/grammars/res_cnn.json', 
])
def test_invalid_models_tolerance(grammar_file):
    parser = Grammar(grammar_file)
    problem = CNNProblem(parser, get_mockup_data_dict())

    num_models = 100
    invalid = 0
    for _ in range(num_models):
        solution = GESolution(parser.dsge_create_solution())
        problem.map_genotype_to_phenotype(solution)
        if solution.phenotype is None: invalid += 1

    assert invalid/num_models <= 0.5