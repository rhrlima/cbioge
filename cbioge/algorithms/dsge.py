import os, glob, time
import datetime as dt

import numpy as np

from cbioge.utils import checkpoint as ckpt
from cbioge.algorithms.solution import GESolution
from cbioge.algorithms.ea import BaseEvolutionaryAlgorithm


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

    def __init__(self, problem, parser, 
        seed=None, 
        pop_size=5, 
        max_evals=10, 
        verbose=False, 
        selection=None, 
        crossover=None, 
        mutation=None, 
        replacement=None):
        super().__init__(problem)

        self.parser = parser

        self.seed = seed
        self.pop_size = pop_size
        self.max_evals = max_evals

        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.replacement = replacement

        self.verbose = verbose

        self.population = None
        self.evals = None
        
        self.all_solutions = []

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

    def evaluate_solution(self, solution: GESolution):
        # skip solutions already executed
        if solution.evaluated:
            if self.verbose:
                curr_time = time.strftime('%x %X')
                print(f'<{curr_time}> [eval] Solution {solution.id} already evaluated. Skipping...')
            return

        # if self.verbose:
        #     curr_time = dt.datetime.today().strftime('%x %X')
        #     print(f'<{curr_time}> [eval] solution {solution.id} started')
        #     print('genotype:', solution.genotype)

        # performs mapping and evaluates taking the time spent
        start_time = dt.datetime.today()
        self.problem.evaluate(solution)
        solution.time = dt.datetime.today() - start_time

        # local changes for checkpoint
        #solution.fitness = fitness
        #solution.phenotype = phenotype
        #solution.evaluated = True
        #solution.params = params

        ckpt.save_solution(solution)

        if self.verbose:
            curr_time = dt.datetime.today().strftime('%x %X')
            print(f'<{curr_time}> [eval] solution {solution.id:4} fit {float(solution.fitness):.2f} gen {solution}')

    def evaluate_population(self, population):
        for s in population:
            # redundancy besides the accept solution
            if self.accept_solution(s):
                self.all_solutions.append(s.genotype[:])
            self.evaluate_solution(s)

    def apply_selection(self):

        return self.selection.execute(self.population)

    def apply_crossover(self, parents):
        if self.crossover is None:
            return parents[0].copy()

        offspring = self.crossover.execute(parents)
        # if offspring is a new solution it has to be mapped and evaluated
        # if offspring != parents[0] and offspring != parents[1]:
        #     offspring.evaluated = False
        #     offspring.phenotype = None
        return offspring

    def apply_mutation(self, offspring):
        if self.mutation is None:
            return offspring.copy()

        muted_solution = self.mutation.execute(offspring)
        # if muted_solution != offspring:
        #     offspring.evaluated = False
        #     offspring.phenotype = None

        return muted_solution

    def apply_replacement(self, offspring_pop):

        return self.replacement.execute(self.population, offspring_pop)

    def accept_solution(self, solution):
        # maintain only unique solutions
        # if solution.genotype in self.all_solutions:
        #    return False
        # keep the addition only in the eval_pop function
        # self.all_solutions.append(solution.genotype[:])
        #return True
        return solution.genotype not in self.all_solutions

    def execute(self, checkpoint=False):

        if checkpoint:
            self.load_state()

        print(len(self.all_solutions))

        if not self.population or not self.evals:
            self.population = self.create_population(self.pop_size)
            ckpt.save_population(self.population)

            self.evaluate_population(self.population)
            self.evals = len(self.population)
            self.save_state()

        print(len(self.all_solutions))

        offspring_pop = ckpt.load_solutions()
        while self.evals < self.max_evals:

            #if offspring_pop == []:
            index = 0
            #while index < self.pop_size:
            while len(offspring_pop) < self.pop_size:
            #for index in range(self.pop_size):
                parents = self.apply_selection()

                offspring = self.apply_crossover(parents)
                offspring = self.apply_mutation(offspring)

                if self.accept_solution(offspring):
                    offspring.id = self.evals + index # check
                    ckpt.save_solution(offspring)
                    offspring_pop.append(offspring)
                    self.all_solutions.append(offspring.genotype[:])
                    index += 1
            # else:
            #     [self.all_solutions.append(s.genotype) for s in offspring_pop]

            print(len(self.all_solutions))
            self.evaluate_population(offspring_pop)

            self.population = self.apply_replacement(offspring_pop)

            self.evals += self.pop_size
            offspring_pop.clear()

            self.save_state()
            self.print_progress()

        return self.population

    def save_state(self):

        data = {
            'evals': self.evals,
            'population': [s.to_json() for s in self.population],
            'unique': self.all_solutions
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
        if 'unique' in data: self.all_solutions = data['unique']

        # TODO temp
        # for s in self.population:
        #     if s.fitness is None:
        #         s.fitness = -1
        
        #self.selection = data['selection']
        #self.crossover = data['crossover']
        #self.mutation = data['mutation']
        #self.replacement = data['replacement']

        print(f'[checkpoint] starting from checkpoint: {data_files[0]}')
        print('CURRENT EVALS:', self.evals)
        print('POP SIZE:', len(self.population))
        print('UNIQUE:', len(self.all_solutions))

    def print_progress(self):
        curr_time = time.strftime('%x %X')
        best = max(self.population, key=lambda x: x.fitness)
        print(f'<{curr_time}> evals: {self.evals}/{self.max_evals}',
              f'best so far: {float(best.fitness):.2f} {best.genotype}')