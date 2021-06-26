from cbioge.algorithms.solution import GESolution
import numpy as np

from .operators import GeneticOperator


class OnePointCrossover(GeneticOperator):

    ''' One Point Crossover combines two solutions into one new by
        combining the first half of the first parent solution and
        the second part of the second parent solution

        cross_rate: chance to apply the operator
    '''

    def __init__(self, cross_rate):
        self.cross_rate = cross_rate

    def __str__(self):
        return 'One Point Crossover'

    def execute(self, parents):
        off1 = parents[0].copy()
        off2 = parents[1].copy()

        if np.random.rand() < self.cross_rate:
            p1 = off1.genotype[:]
            p2 = off2.genotype[:]
            min_len = min(len(p1), len(p2))
            cut = np.random.randint(0, min_len)
            off1.genotype = np.concatenate((p1[:cut], p2[cut:]))
        return [off1]


class DSGECrossover(GeneticOperator):
    ''' One-point crossover adapted to the DSGE encoding.
        The cut point will always be selected acording to the sub-lists and not
        the values inside those lists.

        Ex:
        [ [0, 0], | [0, 0], [0, 0] ] parent 1
        [ [1, 1], | [1, 1], [1, 1] ] parent 2
        [ [0, 0], | [1, 1], [1, 1] ] offspring
    '''

    def __init__(self, cross_rate):
        self.cross_rate = cross_rate

    def __str__(self):
        return 'DSGE Crossover'

    def execute(self, parents, cut=None) -> GESolution:
        # crossover is not applied
        if np.random.rand() > self.cross_rate:
            return parents[0].copy()

        #if np.random.rand() > self.cross_rate:
        #print('[operator] crossover applied')

        # hard copy
        p1 = parents[0].genotype[:]
        p2 = parents[1].genotype[:]
        #min_len = min(len(p1), len(p2))
        #if min_len > 0:

        if cut is None:
            cut = np.random.randint(0, len(p1))

        if p1[:cut] == []: return parents[1].copy()
        if p2[cut:] == []: return parents[0].copy()

        offspring = parents[0].copy()
        offspring.genotype = p1[:cut] + p2[cut:]

        return offspring


class DSGEGeneCrossover(GeneticOperator):
    ''' One-point gene crossover adapted to the DSGE encoding.
        The cut point can include the values inside sub-lists.

        | = possible cut points
        [ | [0, 0], | [0, 0], | [0, 0] | ]
    '''

    def __init__(self, cross_rate):
        self.cross_rate = cross_rate

    def __str__(self):
        return 'DSGE Gene Crossover'

    def execute(self, parents, cut=None):
        off1 = parents[0].copy()
        off2 = parents[1].copy()
        if np.random.rand() < self.cross_rate:
            p1 = off1.genotype[:]
            p2 = off2.genotype[:]
            for i, _ in enumerate(p1):
                min_len = min(len(p1[i]), len(p2[i]))
                if min_len > 0:
                    if cut is None:
                        cut = np.random.randint(0, min_len)
                    off1.genotype[i] = p1[i][:cut] + p2[i][cut:]
        return off1