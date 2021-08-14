import os

import pytest
import numpy as np

from cbioge.grammars import Grammar
from cbioge.algorithms import GESolution
from cbioge.algorithms import PointMutation
from cbioge.algorithms import TerminalMutation
from cbioge.algorithms import NonterminalMutation

def get_mockup_parser():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return Grammar(os.path.join(base_dir, 'assets', 'test_grammar.json'))

@pytest.mark.parametrize("gen, expected, seed", [
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
     [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], 0), 
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
     [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], 42), 
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 
     [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 99)
])
def test_PointMutation(gen, expected, seed):
    
    np.random.seed(seed)

    solution = GESolution(gen)

    parser = get_mockup_parser()
    offspring = PointMutation(parser).execute(solution)

    assert gen != expected
    assert solution.genotype == gen
    assert offspring.genotype == expected
    assert offspring.genotype != gen
    assert offspring == GESolution(expected)