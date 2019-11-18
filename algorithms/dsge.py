import os
import glob
import time
from multiprocessing import Pool
import numpy as np
from utils import checkpoint
from .solutions import GESolution
from .ea import BaseEvolutionaryAlgorithm


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

    def __init__(self, problem, parser):
        super(GrammaticalEvolution, self).__init__(problem)

        self.parser = parser

        self.seed = None
        self.verbose = False

        self.pop_size = 5
        self.max_evals = 100

        self.selection = None
        self.crossover = None
        self.mutation = None
        self.replacement = None

        self.population = None
        self.evals = None

        np.random.seed(seed=self.seed)

    def create_solution(self):

        return GESolution(self.parser.dsge_create_solution())

    def create_population(self, size):
        population = []
        for i in range(size):
            solution = self.create_solution()
            solution.id = i
            population.append(solution)
            #self.save_solution(solution)
        return population

    def evaluate_solution(self, solution):

        curr_time = time.strftime('%x %X')

        if not solution.evaluated:

            if self.verbose:
                print(f'<{curr_time}> [eval] solution {solution.id} started')

            phenotype = self.problem.map_genotype_to_phenotype(solution.genotype)
            loss, acc = self.problem.evaluate(phenotype)

            # local changes for checkpoint
            solution.fitness = acc
            solution.phenotype = phenotype
            solution.evaluated = True

            self.save_solution(solution)

            if self.verbose:
                print(f'<{curr_time}> [eval] solution {solution.id} ended')

            return acc
        else:
            if self.verbose:
                print(f'<{curr_time}> [eval] skipping solution {solution.id}')

            return solution.fitness

    def evaluate_population(self, population):

        for s in population:
            self.evaluate_solution(s)

    def replace(self, population, offspring):

        population += offspring
        population.sort(key=lambda x: x.fitness, reverse=self.maximize)
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

                    offspring.id = self.evals + index  # check
                    self.mutation.execute(offspring)
                    offspring_pop.append(offspring)

                    self.save_solution(offspring)

            self.evaluate_population(offspring_pop)
            self.replace(self.population, offspring_pop)

            self.evals += len(offspring_pop)

            offspring_pop = []

            self.print_progress()

            self.save_state()

        return self.population[0]

    def save_state(self):

        data = {
            'evals': self.evals,
            'population': self.population,
            'selection': self.selection,
            'crossover': self.crossover,
            'mutation': self.mutation,
        }

        folder = checkpoint.ckpt_folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = f'data_{self.evals}.ckpt'
        checkpoint.save_data(data, os.path.join(folder, filename))

        # remove solution files already evaluated
        solution_files = glob.glob(os.path.join(folder, 'solution*.ckpt'))
        [os.remove(file) for file in solution_files]

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

        print(f'[checkpoint] starting from checkpoint: {data_files[0]}')
        self.evals = data['evals']
        self.population = data['population']
        self.selection = data['selection']
        self.crossover = data['crossover']
        self.mutation = data['mutation']

    def save_solution(self, solution):

        folder = checkpoint.ckpt_folder
        if not os.path.exists(folder):
            os.mkdir(folder)

        filename = f'solution{solution.id}.ckpt'

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
