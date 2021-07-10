import pytest

from cbioge.algorithms import GESolution
from cbioge.algorithms.crossover import DSGECrossover, DSGEGeneCrossover

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
def test_DSGECrossover(genA, genB, expected, cut):
    
    parent1 = GESolution(genA)
    parent2 = GESolution(genB)

    mock_cross = parent1.genotype[:cut] + parent2.genotype[cut:]

    offspring = DSGECrossover(1.0).execute([parent1, parent2], cut=cut)

    assert mock_cross == expected
    assert offspring == GESolution(expected)


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
    
    ([[4], [0, 0], [0], [1, 1], [1, 0], [1]], 
     [[5], [0], [0, 0], [1], [1], [1, 1]], 
     [[4], [0], [0, 0], [1], [1], [1, 1]], [1, 0, 1, 0, 0, 1]),
])
def test_DSGEGeneCrossover(genA, genB, expected, cuts):
    
    parent1 = GESolution(genA)
    parent2 = GESolution(genB)

    mock_cross = []
    for i, c in enumerate(cuts):
        mock_cross.append(parent1.genotype[i][:c] + parent2.genotype[i][c:])

    offspring = DSGEGeneCrossover(1.0).execute([parent1, parent2], cuts=cuts)

    assert mock_cross == expected
    assert offspring == GESolution(expected)