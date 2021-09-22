import numpy as np

from .operators import SelectionOperator


class TournamentSelection(SelectionOperator):
    '''Tournament Selection picks N random solutions,
    the best solution among these N is added to list of parents.
    The process is repeated for the number of desired parents.

    # Arguments
    n_parents: the number os parents (default 2)
    t_size: number of solutions selected for the tournament (default 2)
    maximize: if the problem is a maximization problem (default False)'''

    def __init__(self, n_parents=2, t_size=2, maximize=False):
        self.n_parents = n_parents
        self.t_size = t_size
        self.maximize = maximize

        if self.t_size < 1 or self.n_parents < 1:
            raise ValueError('t_size and n_parents must be greater than 0')

    def __str__(self):
        return 'Tournament Selection'

    def _get_n_random(self, population, n_size):
        pool = list()
        while len(pool) < n_size:
            temp = np.random.choice(population)
            if temp not in pool:
                pool.append(temp)
        return pool

    def _get_best(self, options):
        if self.maximize:
            return max(options, key=lambda s: s.fitness)
        return min(options, key=lambda s: s.fitness)

    def execute(self, population):

        if len(population) <= self.t_size:
            raise ValueError('Selection not applied: pop_size <= t_size')

        parents = list()
        while len(parents) < self.n_parents:
            pool = self._get_n_random(population, self.t_size)

            best = self._get_best(pool)

            if best not in parents:
                parents.append(best)

        return parents


class SimilaritySelection(SelectionOperator):

    # def __init__(self, n_parents=2, t_size=2, maximize=False):
    #     self.n_parents = n_parents
    #     self.t_size = t_size
    #     self.maximize = maximize

    # def __str__(self):
    #     return 'Similarity Selection'

    # def execute(self, population):

    #     parents = []
    #     while len(parents) < self.n_parents:
    #         pool = get_n_random(population, self.t_size)
    #         pool.sort(key=lambda s: s.fitness, reverse=self.maximize)

    #     raise NotImplementedError('Not yet implemented.')
    pass