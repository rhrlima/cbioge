import logging
from typing import List
from abc import ABC

import numpy as np

from .solution import Solution

class GeneticOperator(ABC):

    def __init__(self):
        self.logger = logging.getLogger('cbioge')

    def export(self):
        return {'name': self.__str__(), 'config': self.__dict__}


class CrossoverOperator(GeneticOperator):

    def __init__(self, cross_rate: float):
        super().__init__()

        if not 0.0 <= cross_rate <= 1.0:
            raise ValueError(f'Crossover rate must be between 0 and 1: {cross_rate}')

        self.cross_rate = cross_rate

    def execute(self, parents: List[Solution], *args) -> Solution:
        pass


class MutationOperator(GeneticOperator):

    def __init__(self, mut_rate: float):
        super().__init__()

        if not 0.0 <= mut_rate <= 1.0:
            raise ValueError(f'Mutation rate must be between 0 and 1: {mut_rate}')

        self.mut_rate = mut_rate

    def execute(self, solution: Solution) -> Solution:
        raise NotImplementedError('Not implemented yet.')


class ReplacementOperator(GeneticOperator):

    def execute(self,
        population: List[Solution],
        offspring: List[Solution]
    ) -> List[Solution]:
        raise NotImplementedError('Not implemented yet.')


class SelectionOperator(GeneticOperator):

    def execute(self, population: List[Solution]) -> List[Solution]:
        raise NotImplementedError('Not implemented yet.')


class HalfAndHalfOperator(GeneticOperator):
    '''Custom operator

    # Parameters
    - op1: first operator (can be either crossover or mutation)
    - op2: second operator (can be either crossover or mutation)
    - rate: float probability [0, 1] of applying the first operator, the second
    is applied with probability of 1-rate'''

    def __init__(self,
        op1: GeneticOperator,
        op2: GeneticOperator,
        rate: float=0.5
    ):
        super().__init__()

        self.op1 = op1
        self.op2 = op2
        self.rate = rate

    def execute(self, parents: List[Solution]) -> Solution:

        if np.random.rand() < self.rate:
            offspring = self.op1.execute(parents)
        else:
            offspring = self.op2.execute(parents[0].copy())

        return offspring


class HalfAndChoiceOperator(GeneticOperator):

    def __init__(self,
        h_op: GeneticOperator,
        o_ops: List[GeneticOperator],
        h_rate: float=0.5,
        o_rate: List[float]=None
    ):
        super().__init__()

        self.h_op = h_op
        self.o_ops = o_ops
        self.h_rate = h_rate
        self.o_rate = o_rate or [0.5]

    def execute(self, parents: List[Solution]) -> Solution:

        if np.random.rand() < self.h_rate:
            offspring = self.h_op.execute(parents)

        else:
            offspring = parents[0].copy()

            rand = np.random.rand()
            for i, _ in enumerate(self.o_ops):
                if rand < np.sum(self.o_rate[:i+1]):
                    offspring = self.o_ops[i].execute(offspring)
                    break

        return offspring
