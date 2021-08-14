import math

from .operators import ReplacementOperator


class ReplaceWorst(ReplacementOperator):

    def __init__(self, maximize=False):
        self.maximize = maximize

    def execute(self, population, offspring):
        population += offspring

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        return population[:len(offspring)]


class ElitistReplacement(ReplacementOperator):

    def __init__(self, rate=0.1, maximize=False):
        self.rate = rate
        self.maximize = maximize

    def execute(self, population, offspring):

        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
        offspring.sort(key=lambda x: x.fitness, reverse=self.maximize)

        elites = int(math.floor(self.rate * len(population)))

        if elites < 1:
            print('[replace] not applied, elites less than 1')
            return population

        population = population[:elites] + offspring[:-elites]

        return population