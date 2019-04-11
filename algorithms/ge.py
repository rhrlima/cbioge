import os
import glob
import time
from multiprocessing import Pool
import numpy as np
from utils import checkpoint
from .solutions import GESolution
from .ea import BaseEvolutionaryAlgorithm


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

    def __init__(self, problem):
        super(GrammaticalEvolution, self).__init__(problem)

        self.seed = None
        self.verbose = False

        self.pop_size = 5
        self.max_evals = 10

        self.min_genes = 1
        self.max_genes = 10
        self.min_value = 0
        self.max_value = 255

        self.selection = None
        self.crossover = None
        self.mutation = None
        self.prune = None
        self.duplication = None

        self.population = None
        self.evals = None

        np.random.seed(seed=self.seed)

    def create_solution(self, min_size, max_size, min_value, max_value):

        if min_size >= max_size:
            raise ValueError('[create solution] min >= max')

        genes = np.random.randint(min_value, max_value, np.random.randint(
            min_size, max_size))

        return GESolution(genes)

    def create_population(self, size):
        population = []
        for i in range(size):
            solution = self.create_solution(self.min_genes, self.max_genes,
                                            self.min_value, self.max_value)
            solution.id = i
            self.save_solution(solution)
            population.append(solution)
        return population

    def evaluate_solution(self, solution):

        curr_time = time.strftime('%x %X')

        if not solution.evaluated:

            if self.verbose:
                print(f'<{curr_time}> [eval] solution {solution.id} started')

            fitness, model = self.problem.evaluate(solution)

            # local changes for checkpoint
            solution.fitness = fitness
            solution.phenotype = model
            solution.evaluated = True

            self.save_solution(solution)

            if self.verbose:
                print(f'<{curr_time}> [eval] solution {solution.id} ended')

            return fitness, model
        else:
            print(f'<{curr_time}> [eval] skipping solution {solution.id}')
            return solution.fitness, solution.phenotype

    def evaluate_population(self, population):

        pool = Pool(processes=self.max_processes)
        result = pool.map_async(self.evaluate_solution, population)

        pool.close()
        pool.join()

        for sol, res in zip(population, result.get()):
            sol.fitness, sol.phenotype = res
            sol.evaluated = True

        self.population.sort(key=lambda x: x.fitness, reverse=self.maximize)

    def replace(self, population, offspring):

        population += offspring
        for _ in range(len(offspring)):
            population.pop()

    def execute(self, checkpoint=False):

        if checkpoint:
            self.load_state()

        if not self.population or not self.evals:
            self.population = self.create_population(self.pop_size)
            self.evaluate_population(self.population)

            self.evals = len(self.population)

            self.save_state()

        self.print_progress()

        offspring_pop = self.load_solutions()
        while self.evals < self.max_evals:

            if offspring_pop == []:
                for index in range(self.pop_size):
                    parents = self.selection.execute(self.population)
                    offspring = self.crossover.execute(parents)
                    offspring[0].id = self.evals + index  # check
                    self.mutation.execute(offspring)
                    self.prune.execute(offspring)
                    self.duplication.execute(offspring)
                    offspring_pop += offspring

                    self.save_solution(offspring[0])

            self.evaluate_population(offspring_pop)
            self.replace(self.population, offspring_pop)

            self.evals += len(offspring_pop)

            self.print_progress()

            offspring_pop = []

            self.save_state()

        return self.population[0]

    def save_state(self):

        data = {
            'evals': self.evals,
            'population': self.population,
            'selection': self.selection,
            'crossover': self.crossover,
            'mutation': self.mutation,
            'prune': self.prune,
            'duplication': self.duplication
        }

        folder = checkpoint.ckpt_folder
        if not os.path.exists(folder):
            os.mkdir(folder)
        checkpoint.save_data(
            data, os.path.join(folder, f'data_{self.evals}.ckpt'))

        # remove solution files already evaluated
        solution_files = glob.glob(os.path.join(folder, 'solution*.ckpt'))
        for file in solution_files:
            os.remove(file)

    def load_state(self):

        folder = checkpoint.ckpt_folder

        data_files = glob.glob(os.path.join(folder, 'data_*.ckpt'))
        if data_files == []:
            print('[checkpoint] no checkpoint found')
            self.evals = None
            self.population = None
            return

        data_files.sort(key=lambda x: checkpoint.natural_key(x), reverse=True)
        data = checkpoint.load_data(data_files[0])

        self.evals = data['evals']
        self.population = data['population']
        self.selection = data['selection']
        self.crossover = data['crossover']
        self.mutation = data['mutation']
        self.prune = data['prune']
        self.duplication = data['duplication']

    def save_solution(self, solution):

        folder = checkpoint.ckpt_folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = f"solution{solution.id}.ckpt"
        checkpoint.save_data(solution, os.path.join(folder, filename))

    def load_solutions(self):

        folder = checkpoint.ckpt_folder

        solution_files = glob.glob(os.path.join(folder, 'solution*.ckpt'))
        return [checkpoint.load_data(file) for file in solution_files]

    def print_progress(self):
        curr_time = time.strftime('%x %X')
        best = self.population[0].genotype
        best_fit = self.population[0].fitness
        print(f'<{curr_time}> evals: {self.evals}/{self.max_evals}',
              f'best so far: {best} fitness: {best_fit}')
