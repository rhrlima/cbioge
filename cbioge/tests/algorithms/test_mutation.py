import os

import pytest
import numpy as np

from cbioge.grammars import Grammar
from cbioge.algorithms import Solution
from cbioge.algorithms import PointMutation
from cbioge.algorithms import TerminalMutation
from cbioge.algorithms import NonterminalMutation

def get_mockup_parser():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    return Grammar(os.path.join(base_dir, 'assets', 'test_grammar.json'))

@pytest.mark.parametrize('mutation', [
    PointMutation,
    TerminalMutation,
    NonterminalMutation])
@pytest.mark.parametrize('rate', [-1, 2])
def test_invalid_crossover_rate(mutation, rate):

    with pytest.raises(ValueError):
        mutation(get_mockup_parser(), rate)

@pytest.mark.parametrize("gen, expected, seed", [
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
     [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0]], 0),
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
     [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0]], 42),
    ([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
     [[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 99)
])
def test_point_mutation(gen, expected, seed):

    np.random.seed(seed)

    solution = Solution(gen)

    parser = get_mockup_parser()
    offspring = PointMutation(parser, 1).execute(solution)

    assert gen != expected
    assert solution.genotype == gen
    assert offspring.genotype == expected
    assert offspring.genotype != gen
    assert offspring == Solution(expected)
