import numpy as np
import time
from multiprocessing import Pool
from .solutions import GESolution
from .ea import BaseEvolutionaryAlgorithm


class RandomSearch(BaseEvolutionaryAlgorithm):

    MIN_VALUE = 0
    MAX_VALUE = 1
    MIN_SIZE = 1
    MAX_SIZE = 10

    MAX_EVALS = 100

    def __init__(self, problem):

        super(RandomSearch, self).__init__(problem)

    def create_solution(self, min_size, max_size=None, min_value=MIN_VALUE,
                        max_value=MAX_VALUE):
        values = np.random.randint(
            min_value, max_value,
            np.random.randint(min_size, max_size))

        return GESolution(values)

    def evaluate_solution(self, solution):

        return self.problem.evaluate(solution, 1)

    def execute(self):

        best = None
        evals = 0

        while evals < self.MAX_EVALS:

            population = []
            for _ in range(self.MAX_PROCESSES):
                solution = self.create_solution(
                    self.MIN_SIZE, self.MAX_SIZE,
                    self.MIN_VALUE, self.MAX_VALUE)
                population.append(solution)

            pool = Pool(processes=self.MAX_PROCESSES)

            result = pool.map_async(self.evaluate_solution, population)

            pool.close()
            pool.join()

            for solution, result in zip(population, result.get()):
                fit, model = result
                solution.fitness = fit
                solution.phenotype = model
                solution.evaluated = True

            if best:
                population.append(best)
            population.sort(key=lambda x: x.fitness, reverse=self.MAXIMIZE)

            best = population[0].copy(deep=True)
            evals += len(population)

            print('<{}> evals: {}/{} \tbest so far: {}\tfitness: {}'.format(
                time.strftime('%x %X'),
                evals, self.MAX_EVALS,
                best.genotype,
                best.fitness)
            )

        return best
