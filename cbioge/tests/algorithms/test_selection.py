import pytest

from cbioge.algorithms import Solution
from cbioge.algorithms.selection import TournamentSelection

@pytest.mark.parametrize("t_size, n_parents, raises", [
    (2, 2, False), 
    (5, 2, False), ])
def test_tournament_selection(t_size, n_parents, raises):

    selection = TournamentSelection(n_parents=n_parents, t_size=t_size)

    population = [Solution(fitness=v) for v in range(10)]

    parents = selection.execute(population)
    assert len(parents) == n_parents

@pytest.mark.parametrize("t_size, n_parents", [
    (10, 2), 
    (0, 2), 
    (2, 0), 
    (10, 10), ])
def test_tournament_selection_error(t_size, n_parents):

    with pytest.raises(ValueError):
        selection = TournamentSelection(n_parents=n_parents, t_size=t_size)

        population = [Solution(fitness=v) for v in range(10)]

        selection.execute(population)