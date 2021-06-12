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

    def __init__(self, cross_rate):
        self.cross_rate = cross_rate

    def __str__(self):
        return 'DSGE Crossover'

    def execute(self, parents):
        off1 = parents[0].copy()
        off2 = parents[1].copy()

        if np.random.rand() < self.cross_rate:
            #print('[operator] crossover applied')
            p1 = off1.genotype[:]
            p2 = off2.genotype[:]
            min_len = min(len(p1), len(p2))
            if min_len > 0:
                cut = np.random.randint(0, min_len)
                off1.genotype = p1[:cut] + p2[cut:]
            #off2.genotype = p2[:cut] + p1[cut:]
        return off1#[off1, off2]


class DSGEGeneCrossover(GeneticOperator):

    def __init__(self, cross_rate):
        self.cross_rate = cross_rate

    def __str__(self):
        return 'DSGE Gene Crossover'

    def execute(self, parents):
        off1 = parents[0].copy()
        off2 = parents[1].copy()
        if np.random.rand() < self.cross_rate:
            p1 = off1.genotype[:]
            p2 = off2.genotype[:]
            for i, _ in enumerate(p1):
                min_len = min(len(p1[i]), len(p2[i]))
                if min_len > 0:
                    cut = np.random.randint(0, min_len)
                    off1.genotype[i] = p1[i][:cut] + p2[i][cut:]
        return off1