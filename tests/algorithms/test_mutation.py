import pytest

import numpy as np

from cbioge.grammars import Grammar
from cbioge.algorithms import GESolution
from cbioge.algorithms import DSGEMutation
from cbioge.algorithms import DSGETerminalMutation
from cbioge.algorithms import DSGENonterminalMutation

def get_mockup_parser():
    return Grammar('cbioge/assets/grammars/test_grammar.json')

@pytest.mark.parametrize("gen, expected, seed", [
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
     [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], 0), 
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
     [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], 42), 
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
     [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 99)
])
def test_DSGEMutation(gen, expected, seed):
    
    np.random.seed(seed)

    solution = GESolution(gen)

    parser = get_mockup_parser()
    offspring = DSGEMutation(1.0, parser).execute(solution)

    assert gen != expected
    assert solution.genotype == gen
    assert offspring.genotype == expected
    assert offspring.genotype != gen
    assert offspring == GESolution(expected)