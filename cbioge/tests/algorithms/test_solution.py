import pytest

from cbioge.algorithms import Solution

def test_create_solution_from_list():
    solution = Solution([[0,0],[0,0,0],[0]])
    assert solution.genotype == [[0,0],[0,0,0],[0]]

def test_create_solution_from_json():
    json_string = {
        'id': None,
        'genotype': [[1, 1], [1, 1, 1], [1]],
        'phenotype': {'a': 123},
        'fitness': 200,
        'evaluated': False,
        'data': {
            'time': None,
            'params': None
        }
    }
    solution = Solution.from_json(json_string)
    assert solution.to_json() == json_string
    assert solution.fitness == 200
    assert json_string['fitness'] == 200

def test_create_solution_from_incomplete_json():
    json_string = {
        'id': None,
        'genotype': [[1, 1], [1, 1, 1], [1]],
        'phenotype': None,
        'evaluated': False,
        'data': {
            'params': None
        }
    }
    solution = Solution.from_json(json_string)
    json_data = solution.to_json()
    assert json_data != json_string
    assert len(json_data.keys()) > len(json_string.keys())
    assert 'fitness' not in json_string
    assert 'fitness' in json_data

def test_create_solution_from_incomplete_json2():
    json_string = {
        'id': None,
        'genotype': [],
        'phenotype': None,
        'fitness': -1,
        'evaluated': False,
        'data': {}
    }
    solution = Solution.from_json(None)
    assert solution.to_json() == json_string

def test_create_empty_solution():
    json_string = {
        'id': None,
        'genotype': [],
        'phenotype': None,
        'fitness': -1,
        'evaluated': False,
        'data': {}
    }
    solution = Solution()
    assert solution.to_json() == json_string


@pytest.mark.parametrize("gen_a, gen_b, expected", [
    ([[0,0],[0,0,0],[0]], [[0,0],[0,0,0],[0]], True),
    ([[0,0],[0,0,0],[0]], [[0,0],[0,0,0],[1]], False),
    ([[0,0],[0,0,0],[0]], 'copy', True),
    ([[0,0],[0,0,0],[0]], 'json', True),
    ([[0,0],[0,0,0],[0]], None, False),
    ([[0,0],[0,0,0],[0]], [], False)])
def test_eq_override(gen_a, gen_b, expected):
    obj_a = Solution(gen_a)
    if gen_b == 'copy':
        obj_b = obj_a.copy()
    elif gen_b == 'json':
        obj_b = Solution.from_json(obj_a.to_json())
    else:
        obj_b = Solution(gen_b)

    assert (obj_a == obj_b) == expected

def test_modified_copy():
    sol_a = Solution([[0], [0], [0], [0]])
    sol_b = sol_a.copy()

    assert sol_a == sol_b
    assert sol_a.to_json() == sol_b.to_json()

    sol_b.genotype[0][0] = 1

    assert sol_a != sol_b
    assert sol_a.to_json() != sol_b.to_json()

def test_modified_deepcopy():

    sol_a = Solution([[0], [0], [0], [0]])
    sol_b = sol_a.copy(deep=True)

    assert sol_a == sol_b
    assert sol_a.to_json() == sol_b.to_json()

    sol_b.genotype[0][0] = 1
    sol_b.fitness = 999

    assert sol_a != sol_b
    assert sol_a.to_json() != sol_b.to_json()

def test_solution_in_list():

    aux_list = [Solution([[0], [0], [0], [0]])]
    new_solution = Solution([[0], [0], [0], [0]])

    ref_solution = aux_list[0]
    cpy_solution = aux_list[0].copy()
    dpy_solution = aux_list[0].copy(deep=True)

    mod1_solution = aux_list[0].copy(deep=True)
    mod2_solution = aux_list[0].copy(deep=True)

    mod1_solution.genotype[0] = 1
    mod2_solution.evaluated = True

    assert new_solution in aux_list
    assert ref_solution in aux_list
    assert cpy_solution in aux_list
    assert dpy_solution in aux_list
    assert mod1_solution not in aux_list
    assert mod2_solution not in aux_list
