import os
import glob
import time
import datetime as dt

import numpy as np

from multiprocessing import Pool

from utils import checkpoint as ckpt
from .solution import GESolution
from .ea import BaseEvolutionaryAlgorithm


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

    def __init__(self, problem, parser):
        super(GrammaticalEvolution, self).__init__(problem)

        self.parser = parser

        self.seed = None
        self.pop_size = 5
        self.max_evals = 100
        self.training = True

        self.selection = None
        self.crossover = None
        self.mutation = None
        self.replacement = None

        self.population = None
        self.evals = None

        self.verbose = False

        np.random.seed(seed=self.seed)

    def create_solution(self):

        return GESolution(self.parser.dsge_create_solution())

    def create_population(self, size):
        population = []
        for i in range(size):
            solution = self.create_solution()
            solution.id = i
            population.append(solution)
        return population

    def evaluate_solution(self, solution):

        if solution.evaluated:
            if self.verbose:
                curr_time = time.strftime('%x %X')
                print(f'<{curr_time}> [eval] skipping solution {solution.id}. Already evaluated')
            return

        if self.verbose:
            curr_time = dt.datetime.today().strftime('%x %X')
            print(f'<{curr_time}> [eval] solution {solution.id} started')
            print('genotype:', solution.genotype)

        phenotype = self.problem.map_genotype_to_phenotype(solution.genotype)

        start_time = dt.datetime.today()
        scores, params = self.problem.evaluate(phenotype)
        end_time = dt.datetime.today()

        # scores:
        # 0: loss
        # 1: accuracy
        # 2..: others
        fitness = scores[1]

        # local changes for checkpoint
        solution.fitness = fitness
        solution.phenotype = phenotype
        solution.evaluated = True
        solution.time = end_time - start_time
        solution.params = params

        ckpt.save_solution(solution)

        if self.verbose:
            curr_time = dt.datetime.today().strftime('%x %X')
            print('fitness:', solution.fitness)
            print(f'<{curr_time}> [eval] solution {solution.id} ended')

    def evaluate_population(self, population):

        for s in population:
            self.evaluate_solution(s)

    def apply_selection(self):

        return self.selection.execute(self.population)

    def apply_crossover(self, parents):        
        if self.crossover is not None:
            return self.crossover.execute(parents)
        return parents[0].copy()

    def apply_mutation(self, offspring):
        if self.mutation is not None:
            return self.mutation.execute(offspring)
        return offspring.copy()

    def apply_replacement(self, offspring_pop):

        return self.replacement.execute(self.population, offspring_pop)

    def execute(self, checkpoint=False):

        if checkpoint:
            self.load_state()

        if not self.population or not self.evals:
            self.population = self.create_population(self.pop_size)
            ckpt.save_population(self.population)

            self.evaluate_population(self.population)
            self.evals = len(self.population)
            self.save_state()

        self.print_progress()

        offspring_pop = ckpt.load_solutions()
        while self.evals < self.max_evals:

            if offspring_pop == []:
                for index in range(self.pop_size):
                    parents = self.apply_selection()

                    offspring = self.apply_crossover(parents)
                    offspring.id = self.evals + index # check
                    
                    offspring = self.apply_mutation(offspring)

                    ckpt.save_solution(offspring)

                    offspring_pop.append(offspring)

            self.evaluate_population(offspring_pop)

            self.population = self.apply_replacement(offspring_pop)

            self.evals += len(offspring_pop)
            offspring_pop.clear()

            self.save_state()
            self.print_progress()

        return self.population

    def save_state(self):

        data = {
            'evals': self.evals,
            'population': [s.to_json() for s in self.population],
            #'selection': self.selection,
            #'crossover': self.crossover,
            #'mutation': self.mutation,
            #'replacement': self.replacement
        }

        filename = f'data_{self.evals}.ckpt'
        saved = ckpt.save_data(data, filename)

        # remove solution files already evaluated if data ckpt exists
        if saved: ckpt.delete_solution_checkpoints('solution_*.ckpt')
        
    def load_state(self):

        folder = ckpt.ckpt_folder

        data_files = glob.glob(os.path.join(folder, 'data_*.ckpt'))
        if data_files == []:
            print('[checkpoint] no checkpoint found')
            self.evals = None
            self.population = None
            return

        data_files.sort(key=lambda x: ckpt.natural_key(x), reverse=True)
        data = ckpt.load_data(data_files[0])

        
        self.evals = data['evals']
        self.population = [GESolution(json_data=json_data) for json_data in data['population']]

        # temp
        for s in self.population:
            if s.fitness is None:
                s.fitness = -1
        
        #self.selection = data['selection']
        #self.crossover = data['crossover']
        #self.mutation = data['mutation']
        #self.replacement = data['replacement']

        print(f'[checkpoint] starting from checkpoint: {data_files[0]}')
        print('Evals:', self.evals)
        print('Population:', len(self.population))

    def print_progress(self):
        curr_time = time.strftime('%x %X')
        best = self.population[0].genotype
        best_fit = self.population[0].fitness
        print(f'<{curr_time}> evals: {self.evals}/{self.max_evals}',
              f'best so far: {best} fitness: {best_fit}')
