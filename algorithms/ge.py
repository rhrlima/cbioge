import os
import glob
import time
from multiprocessing import Pool
import numpy as np
from utils import checkpoint
from .solutions import GESolution
from .ea import BaseEvolutionaryAlgorithm


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

    DEBUG = False
    verbose = False

    def __init__(self, problem):
        super(GrammaticalEvolution, self).__init__(problem)

        self.seed = None

        self.POP_SIZE = 5
        self.MAX_EVALS = 10

        self.MIN_GENES = 1
        self.MAX_GENES = 10
        self.MIN_VALUE = 0
        self.MAX_VALUE = 255

        self.selection = None
        self.crossover = None
        self.mutation = None
        self.prune = None
        self.duplication = None

        self.population = None
        self.evals = None

    def create_solution(self, min_size, max_size, min_value, max_value):
        if not max_size:
            max_size = min_size
            min_size = 0

        if min_size >= max_size:
            raise ValueError('[create solution] min >= max')

        genes = np.random.randint(min_value, max_value, np.random.randint(
            min_size, max_size))

        return GESolution(genes)

    def create_population(self, size):
        population = []
        for i in range(size):
            solution = self.create_solution(self.MIN_GENES, self.MAX_GENES,
                                            self.MIN_VALUE, self.MAX_VALUE)
            solution.id = i
            population.append(solution)
        return population

    def evaluate_solution(self, solution):

        print('[eval] solution', solution.id, 'started')

        if not solution.evaluated:
            fitness, model = self.problem.evaluate(solution)
            self.save_solution(solution)
        else:
            print(f'[eval] solution {solution.id} already evaluated')
            fitness = solution.fitness
            model = solution.model

        print('[eval] solution', solution.id, 'ended')

        return fitness, model

    def evaluate_population(self, population):

        pool = Pool(processes=self.max_processes)
        result = pool.map_async(self.evaluate_solution, population)

        pool.close()
        pool.join()

        for sol, res in zip(population, result.get()):
            fit, model = res
            sol.fitness = fit
            sol.phenotype = model
            sol.evaluated = True

    def replace(self, population, offspring):

        population += offspring
        population.sort(key=lambda x: x.fitness, reverse=self.maximize)

        for _ in range(len(offspring)):
            population.pop()

    def execute(self, checkpoint=False):

        if checkpoint:
            print('[execute] starting from checkpoint')
            self.load_state()

        if not self.population or not self.evals:
            print('[execute] starting from scratch')
            self.population = self.create_population(self.POP_SIZE)
            print('## evaluate')
            self.evaluate_population(self.population)
            self.population.sort(key=lambda x: x.fitness,
                                 reverse=self.maximize)
            self.evals = len(self.population)

            if self.DEBUG:
                for i, p in enumerate(self.population):
                    print(i, p.fitness, p)

            self.save_state()

        print('<{}> evals: {}/{} \tbest so far: {}\tfitness: {}'.format(
            time.strftime('%x %X'),
            self.evals, self.MAX_EVALS,
            self.population[0].genotype,
            self.population[0].fitness))

        while self.evals < self.MAX_EVALS:

            parents = self.selection.execute(self.population)
            offspring_pop = self.load_solutions()

            if offspring_pop == []:
                for index in range(self.POP_SIZE):
                    offspring = self.crossover.execute(parents)
                    offspring[0].id = self.evals + index  # check
                    self.mutation.execute(offspring)
                    self.prune.execute(offspring)
                    self.duplication.execute(offspring)
                    offspring_pop += offspring

            self.evaluate_population(offspring_pop)
            self.replace(self.population, offspring_pop)

            self.evals += len(offspring_pop)

            if self.DEBUG:
                for i, p in enumerate(self.population):
                    print(i, p.fitness, p)

            print('<{}> evals: {}/{} \tbest so far: {}\tfitness: {}'.format(
                time.strftime('%x %X'),
                self.evals, self.MAX_EVALS,
                self.population[0].genotype,
                self.population[0].fitness))

            self.save_state()

        return self.population[0]

    def save_state(self):

        data = {
            'evals': self.evals,
            'population': self.population,
        }

        folder = checkpoint.ckpt_folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        checkpoint.save_data(
            data, os.path.join(folder, f'data_{self.evals}.ckpt'))

    def load_state(self):

        folder = checkpoint.ckpt_folder

        data_files = glob.glob(os.path.join(folder, 'data_*'))
        if data_files == []:
            print('[checkpoint] no data file found')
            self.evals = None
            self.population = None
            return

        data_files.sort(key=lambda x: checkpoint.natural_key(x), reverse=True)
        data = checkpoint.load_data(data_files[0])

        self.evals = data['evals']
        self.population = data['population']

    def save_solution(self, solution):

        folder = checkpoint.ckpt_folder

        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = f"solution{solution.id}.ckpt"
        checkpoint.save_data(solution, os.path.join(folder, filename))

    def load_solutions(self):

        folder = checkpoint.ckpt_folder

        solution_files = glob.glob(os.path.join(folder, 'solution*.ckpt'))
        if solution_files == []:
            print('[checkpoint] no solution files found')
            return []

        print('[checkpoint] solution files found')
        return [checkpoint.load_data(file) for file in solution_files]
