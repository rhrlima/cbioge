from typing import List

import numpy as np

from .solution import Solution
from .operators import CrossoverOperator


class OnePointCrossover(CrossoverOperator):
    '''One-point crossover adapted to the DSGE encoding.\n
    The cut point will always be selected acording to the sub-lists and not
    the values inside those lists.

    Ex: cut = 1\n
    [ [0, 0], | [0, 0], [0, 0] ] parent 1\n
    [ [1, 1], | [1, 1], [1, 1] ] parent 2\n
    [ [0, 0], | [1, 1], [1, 1] ] offspring'''

    def __init__(self, cross_rate: float=1.0):
        super().__init__(cross_rate)

    def __str__(self):
        return 'One Point Crossover'

    def execute(self, parents: List[Solution], cut: int=None) -> Solution:

        # crossover is not applied
        if np.random.rand() > self.cross_rate:
            return parents[0].copy()

        gen1 = parents[0].genotype[:]
        gen2 = parents[1].genotype[:]

        # if cut is not defined, picks a random one
        if cut is None: cut = np.random.randint(0, len(gen1))

        return Solution(gen1[:cut] + gen2[cut:])


class TwoPointsCrossover(CrossoverOperator):
    pass


class GeneCrossover(CrossoverOperator):
    ''' One-point gene crossover adapted to the DSGE encoding.\n
        An individual cut will happen for each sub-list of the solution.

        Ex: cuts = [0, 1, 2]\n
        [ [|0, 0], [0,| 0], [0, 0|] ] parent 1\n
        [ [|1, 1], [1,| 1], [1, 1|] ] parent 2\n
        [ [|1, 1], [0,| 1], [0, 0|] ] offspring
    '''

    def __init__(self, cross_rate: float=1.0):
        super().__init__(cross_rate)

    def __str__(self):
        return 'Gene Crossover'

    def execute(self, parents: List[Solution], cuts: List[int]=None) -> Solution:

        # crossover is not applied
        if np.random.rand() > self.cross_rate:
            return parents[0].copy()

        gen1 = parents[0].genotype[:]
        gen2 = parents[1].genotype[:]

        new_gen = list()
        for c_idx in range(len(parents[0].genotype)):

            if cuts is None:
                min_len = min(len(gen1[c_idx]), len(gen2[c_idx]))
                cut = np.random.randint(0, min_len)
            else:
                cut = cuts[c_idx]

            new_gen.append(gen1[c_idx][:cut] + gen2[c_idx][cut:])

        return Solution(new_gen)