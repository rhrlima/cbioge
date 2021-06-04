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

def test_compare_solutions_from_copy():

    solution = GESolution([[0,0],[0,0,0],[0]])
    json_string = solution.to_json()

    solution_copy = solution.copy()
    solution_json = GESolution(json_data=json_string)

    assert (solution is solution_copy) == False
    assert (solution == solution_copy) == False
    
    assert (solution is solution_json) == False
    assert (solution == solution_json) == False

    assert (solution_copy is solution_json) == False
    assert (solution_copy == solution_json) == False
