import math

from .operators import ReplacementOperator


class ReplaceWorst(ReplacementOperator):
    '''Replaces solutions maintaining only the best individuals among the 
    population and offspring.'''

    def __init__(self, maximize=False):
        self.maximize = maximize

    def execute(self, population, offspring):
        population += offspring

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        return population[:len(offspring)]


class ElitistReplacement(ReplacementOperator):
    '''Replace the parent population by the offspring, maintaining a # of elites.
    The best offpsring are selected for the replacement.'''

    def __init__(self, rate=0.1, maximize=False):
        super().__init__()
        self.rate = rate
        self.maximize = maximize

    def execute(self, population, offspring):

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        offspring.sort(key=lambda x: x.fitness, reverse=not self.maximize)

        elites = max(int(math.floor(self.rate * len(population))), 0)

        if elites == 0:
            # full replacement
            self.logger.warning(f'{self} not applied. Number of elites less than 1.')

        return population[:elites] + offspring[elites:]