import pytest

from cbioge.algorithms import Solution
from cbioge.algorithms.operators import CrossoverOperator
from cbioge.algorithms.crossover import OnePointCrossover, GeneCrossover


@pytest.mark.parametrize('crossover', [
    OnePointCrossover,
    GeneCrossover])
@pytest.mark.parametrize('rate', [-1, 2])
def test_invalid_crossover_rate(crossover, rate):

    with pytest.raises(ValueError):
        crossover(rate)

@pytest.mark.parametrize('crossover', [
    CrossoverOperator,
    OnePointCrossover,
    GeneCrossover])
def test_liskov_principle_crossover(crossover):

    parent_a = parent_b = Solution([])
    crossover(1.0).execute([parent_a, parent_b], None)

@pytest.mark.parametrize("gen_a, gen_b, expected, cut", [
    ([[4], [0, 0], [0], [1, 1], [1, 0], [1]],
     [[5], [0], [0, 0], [1], [1], [1, 1]],
     [[4], [0], [0, 0], [1], [1], [1, 1]], 1),

    ([[0], [0], [0], [0], [0], [0]],
     [[1], [1], [1], [1], [1], [1]],
     [[0], [0], [0], [1], [1], [1]], 3),

    ([[0], [0], [0], [0]],
     [[1], [1], [1], [1]],
     [[1], [1], [1], [1]], 0),

    ([[0], [0], [0], [0]],
     [[2], [2], [2], [2]],
     [[0], [0], [0], [0]], 4)])
def test_one_point_crossover(gen_a, gen_b, expected, cut):

    parent1 = Solution(gen_a)
    parent2 = Solution(gen_b)

    mock_cross = parent1.genotype[:cut] + parent2.genotype[cut:]

    offspring = OnePointCrossover(1.0).execute([parent1, parent2], cut=cut)

    print(mock_cross, offspring)

    assert mock_cross == expected
    assert offspring == Solution(expected)


@pytest.mark.parametrize("gen_a, gen_b, expected, cuts", [
    ([[0, 0], [0, 0], [0, 0]],
     [[1, 1], [1, 1], [1, 1]],
     [[0, 1], [0, 1], [0, 1]], [1, 1, 1]),

    ([[0, 0], [0, 0], [0, 0]],
     [[1, 1], [1, 1], [1, 1]],
     [[1, 1], [1, 1], [1, 1]], [0, 0, 0]),

    ([[0, 0], [0, 0], [0, 0]],
     [[1, 1], [1, 1], [1, 1]],
     [[0, 0], [0, 0], [0, 0]], [2, 2, 2]),

    ([[0, 0], [0, 0], [0, 0]],
     [[1, 1], [1, 1], [1, 1]],
     [[0, 0], [0, 1], [1, 1]], [2, 1, 0]),

    ([[4], [0, 0], [0], [1, 1], [1, 0], [1]],
     [[5], [0], [0, 0], [1], [1], [1, 1]],
     [[4], [0], [0, 0], [1], [1], [1, 1]], [1, 0, 1, 0, 0, 1]),])
def test_gene_crossover(gen_a, gen_b, expected, cuts):

    parent1 = Solution(gen_a)
    parent2 = Solution(gen_b)

    mock_cross = []
    for i, c in enumerate(cuts):
        mock_cross.append(parent1.genotype[i][:c] + parent2.genotype[i][c:])

    offspring = GeneCrossover(1.0).execute([parent1, parent2], cuts=cuts)

    assert mock_cross == expected
    assert offspring == Solution(expected)
