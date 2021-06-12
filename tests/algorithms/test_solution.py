import pytest

from cbioge.algorithms.solution import GESolution

def test_create_solution_from_list():
    solution = GESolution([[0,0],[0,0,0],[0]])
    assert solution.genotype == [[0,0],[0,0,0],[0]]

def test_create_solutino_from_json():
    json_string = {
        'id': None, 
        'genotype': [[1, 1], [1, 1, 1], [1]], 
        'phenotype': None, 
        'fitness': -1, 
        'evaluated': False, 
        'time': None, 
        'params': None
    }
    solution = GESolution(json_data=json_string)
    assert solution.to_json() == json_string

@pytest.mark.parametrize("genA, genB, expected", [
    ([[0,0],[0,0,0],[0]], [[0,0],[0,0,0],[0]], True),
    ([[0,0],[0,0,0],[0]], [[0,0],[0,0,0],[1]], False),
    ([[0,0],[0,0,0],[0]], 'copy', True),
    ([[0,0],[0,0,0],[0]], 'json', True),
    ([[0,0],[0,0,0],[0]], None, False),
    ([[0,0],[0,0,0],[0]], [], False)])
def test_is_equal2(genA, genB, expected):
    objA = GESolution(genA)
    if genB == 'copy':
        objB = objA.copy()
    elif genB == 'json':
        objB = GESolution(json_data=objA.to_json()) 
    else:
        objB = GESolution(genB)
    assert (objA == objB) == expected
