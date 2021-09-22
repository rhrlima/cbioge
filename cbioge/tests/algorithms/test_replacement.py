from cbioge.algorithms import Solution
from cbioge.algorithms.replacement import ReplaceWorst, ElitistReplacement

import pytest

def test_replace_worst():

    replacement = ReplaceWorst(maximize=True)

    population = [Solution(fitness=f) for f in range(0, 20, 2)]
    offspring = [Solution(fitness=f) for f in range(1, 21, 2)]
    expected = [Solution(fitness=f) for f in range(10, 20)]

    result = replacement.execute(population, offspring)

    expected.sort(key=lambda x: x.fitness, reverse=True)
    for i in range(len(result)):
        assert result[i].fitness == expected[i].fitness


@pytest.mark.parametrize("rate, expected_fit", [
    (0.0, range(1, 21, 2)), 
    (1.0, range(0, 20, 2)), 
    (0.5, range(10, 21)), 
    (0.2, [5, 7, 9, 11, 13, 15, 16, 17, 18, 19]), ])
def test_replace_with_elitism(rate, expected_fit):

    replacement = ElitistReplacement(rate, maximize=True)

    population = [Solution(fitness=f) for f in range(0, 20, 2)]
    offspring = [Solution(fitness=f) for f in range(1, 21, 2)]
    expected = [Solution(fitness=f) for f in expected_fit]

    result = replacement.execute(population, offspring)

    expected.sort(key=lambda x: x.fitness)
    result.sort(key=lambda x: x.fitness)

    for i in range(len(result)):
        assert result[i].fitness == expected[i].fitness