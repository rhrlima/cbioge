import math
from typing import List

from .solution import Solution
from .operators import ReplacementOperator


class ReplaceWorst(ReplacementOperator):
    '''Replaces solutions maintaining only the best individuals among the
    population and offspring.'''

    def __init__(self, maximize: bool=False):
        super().__init__()

        self.maximize = maximize

    def execute(self,
        population: List[Solution],
        offspring: List[Solution]
    ) -> List[Solution]:
        population += offspring

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        return population[:len(offspring)]


class ElitistReplacement(ReplacementOperator):
    '''Replace the parent population by the offspring, maintaining a # of elites.
    The best offpsring are selected for the replacement.'''

    def __init__(self, rate: float=0.1, maximize: bool=False):
        super().__init__()

        self.rate = rate
        self.maximize = maximize

    def execute(self,
        population: List[Solution],
        offspring: List[Solution]
    ) -> List[Solution]:

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        offspring.sort(key=lambda x: x.fitness, reverse=not self.maximize)

        elites = max(int(math.floor(self.rate * len(population))), 0)

        if elites == 0:
            # full replacement
            warn_text = f'{self} not applied. Number of elites less than 1.'
            self.logger.warning(warn_text)

        return population[:elites] + offspring[elites:]
