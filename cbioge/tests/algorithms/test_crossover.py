import pytest

from cbioge.algorithms import Solution
from cbioge.algorithms.crossover import OnePointCrossover, GeneCrossover

@pytest.mark.parametrize('crossover', [
    OnePointCrossover, 
    GeneCrossover])
def test_negative_crossover_rate(crossover):

    invalid_rate = -1

    with pytest.raises(ValueError):
        crossover(invalid_rate)

@pytest.mark.parametrize('crossover', [
    OnePointCrossover, 
    GeneCrossover])
def test_invalid_crossover_rate(crossover):

    invalid_rate = 2

    with pytest.raises(ValueError):
        crossover(invalid_rate)
    

@pytest.mark.parametrize("genA, genB, expected, cut", [
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
def test_OnePointCrossover(genA, genB, expected, cut):
    
    parent1 = Solution(genA)
    parent2 = Solution(genB)

    mock_cross = parent1.genotype[:cut] + parent2.genotype[cut:]

    offspring = OnePointCrossover().execute([parent1, parent2], cut=cut)

    assert mock_cross == expected
    assert offspring == Solution(expected)


@pytest.mark.parametrize("genA, genB, expected, cuts", [
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
def test_GeneCrossover(genA, genB, expected, cuts):
    
    parent1 = Solution(genA)
    parent2 = Solution(genB)

    mock_cross = []
    for i, c in enumerate(cuts):
        mock_cross.append(parent1.genotype[i][:c] + parent2.genotype[i][c:])

    offspring = GeneCrossover().execute([parent1, parent2], cuts=cuts)

    assert mock_cross == expected
    assert offspring == Solution(expected)