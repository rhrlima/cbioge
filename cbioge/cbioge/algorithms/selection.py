import numpy as np

from cbiogeevolution.algorithms.operators import GeneticOperator


def get_n_random(population, n_size):
    pool = []
    while len(pool) < n_size:
        temp = np.random.choice(population)
        if temp not in pool:
            pool.append(temp)
    return pool


class TournamentSelection(GeneticOperator):

    ''' Tournament Selection picks N random solutions,
        the best solution among these N is added to list of parents.
        The process is repeated for the number of desired parents.

        n_parents: the number os parents (default 2)
        t_size: number of solutions selected for the tournament (default 2)
        maximize: if the problem is a maximization problem (default False)
    '''

    def __init__(self, n_parents=2, t_size=2, maximize=False):
        self.n_parents = n_parents
        self.t_size = t_size
        self.maximize = maximize

    def __str__(self):
        return 'Tournament Selection'

    def execute(self, population):

        if len(population) <= self.t_size:
            raise ValueError('population size <= tournament size')

        parents = []
        while len(parents) < self.n_parents:
            pool = get_n_random(population, self.t_size)
            pool.sort(key=lambda s: s.fitness, reverse=self.maximize)
            if pool[0] not in parents:
                parents.append(pool[0])
        return parents


class SimilaritySelection(GeneticOperator):

    def __init__(self, n_parents=2, t_size=2, maximize=False):
        self.n_parents = n_parents
        self.t_size = t_size
        self.maximize = maximize

    def __str__(self):
        return 'Similarity Selection'

    def execute(self, population):

        parents = []
        while len(parents) < self.n_parents:
            pool = get_n_random(population, self.t_size)
            pool.sort(key=lambda s: s.fitness, reverse=self.maximize)
        return parents