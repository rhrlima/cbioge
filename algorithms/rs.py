import numpy as np
import time
from multiprocessing import Pool
from .solutions import GESolution
from .ea import BaseEvolutionaryAlgorithm


class RandomSearch(BaseEvolutionaryAlgorithm):

    def __init__(self, problem):
        super(RandomSearch, self).__init__(problem)

        self.seed = None

        self.min_value = 0
        self.max_value = 1
        self.min_size = 1
        self.max_size = 10
        self.max_evals = 100

        self.best = None
        self.evals = 0

        np.random.seed(seed=self.seed)

    def create_solution(self, min_size, max_size, min_value, max_value):
        values = np.random.randint(min_value, max_value, np.random.randint(
            min_size, max_size))

        return GESolution(values)

    def evaluate_solution(self, solution):

        if not solution.evaluated:
            return self.problem.evaluate(solution)
        else:
            return solution.fitness, solution.phenotype

    def execute(self):

        while self.evals < self.max_evals:

            population = []
            for _ in range(self.max_processes):
                solution = self.create_solution(
                    self.min_size, self.max_size,
                    self.min_value, self.max_value)
                population.append(solution)

            pool = Pool(processes=self.max_processes)
            result = pool.map_async(self.evaluate_solution, population)

            pool.close()
            pool.join()

            for solution, result in zip(population, result.get()):
                solution.fitness, solution.phenotype = result
                solution.evaluated = True

            if self.best:
                population.append(self.best)
            population.sort(key=lambda x: x.fitness, reverse=self.maximize)

            self.best = population[0].copy(deep=True)
            self.evals += len(self.max_processes)

        return self.best

    def print_progress(self):
        curr_time = time.strftime('%x %X')
        print(f'<{curr_time}> evals: {self.evals}/{self.max_evals} ',
              f'best so far: {self.best} fitness: {self.best.fitness}')
