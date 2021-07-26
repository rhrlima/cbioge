import logging

import numpy as np

from .solution import GESolution
from .operators import ReplaceWorst
from .selection import TournamentSelection
from ..utils import checkpoint as ckpt


class BaseEvolutionaryAlgorithm:

    def __init__(self, problem, 
        seed=None, 
        pop_size=5, 
        max_evals=10, 
        verbose=False, 
        **kwargs):

        self.problem = problem
        self.maximize = False

        self.seed = seed
        self.pop_size = pop_size
        self.max_evals = max_evals

        # default operators
        self.selection = TournamentSelection(2, 2, maximize=self.maximize)
        self.replacement = ReplaceWorst(maximize=self.maximize)
        self.crossover = None
        self.mutation = None

        if 'selection' in kwargs:
            self.selection = kwargs['selection']
            kwargs.pop('selection')
        
        if 'replacement' in kwargs:
            self.replacement = kwargs['replacement']
            kwargs.pop('replacement')

        if 'crossover' in kwargs:
            self.crossover = kwargs['crossover']
            kwargs.pop('crossover')

        if 'mutation' in kwargs:
            self.mutation = kwargs['mutation']
            kwargs.pop('mutation')

        self.verbose = verbose

        np.random.seed(seed=self.seed)
        self.logger = logging.getLogger('cbioge')

    def create_solution(self):
        raise NotImplementedError('Not implemented yet.')

    def evaluate_solution(self, solution):
        raise NotImplementedError('Not implemented yet.')

    def apply_selection(self):
        return self.selection.execute(self.population)

    def apply_crossover(self, parents):
        if self.crossover is None:
            return parents[0].copy()
        return self.crossover.execute(parents)

    def apply_mutation(self, offspring):
        if self.mutation is None:
            return offspring.copy()
        return self.mutation.execute(offspring)

    def apply_replacement(self, offspring_pop):
        return self.replacement.execute(self.population, offspring_pop)

    def execute(self):
        raise NotImplementedError('Not implemented yet.')

    def print_progress(self):
        best = max(self.population, key=lambda x: x.fitness)
        log_text = f'evals: {self.evals}/{self.max_evals} best so far: {float(best.fitness): .2f} gen: {best.genotype}'
        self.logger.info(log_text)

    def save_solution(self, solution):
        json_solution = solution.to_json()
        filename = ckpt.solution_name.format(solution.id)
        ckpt.save_data(json_solution, filename)

    def load_solution(self, solution_id):
        try:
            return GESolution(
                json_data=ckpt.load_data(
                    ckpt.solution_name.format(solution_id)
                ))
        except Exception as e:
            return None