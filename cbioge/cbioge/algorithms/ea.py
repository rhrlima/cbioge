import logging

import numpy as np

from .solution import Solution
from .replacement import ReplaceWorst
from .selection import TournamentSelection
from ..utils import checkpoint as ckpt


class BaseEvolutionaryAlgorithm:
    '''Base structure for the evolutionary search.

    It defines basic operations for creating solutions, applying operators, 
    logging, and saving the state of the algorithm.'''

    def __init__(self, problem, 
        seed=None, 
        pop_size=5, 
        max_evals=10, 
        verbose=False, 
        selection=TournamentSelection(2, 2, maximize=True), 
        replacement=ReplaceWorst(maximize=True), 
        crossover=None, 
        mutation=None):

        self.problem = problem

        self.seed = seed
        self.pop_size = pop_size
        self.max_evals = max_evals

        self.selection = selection
        self.replacement = replacement
        self.crossover = crossover
        self.mutation = mutation

        self.verbose = verbose

        self.evals = 0
        self.population = list()

        np.random.seed(seed=self.seed)
        self.logger = logging.getLogger('cbioge')

    def create_solution(self):
        '''Creates a solution following the encoding defined by the grammar-system
        in DSGE.
        
        Ex: [[a, b, c], [d, e, f], ...] where a-f represents integer values.'''
        return Solution(self.problem.parser.create_solution())

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
            return Solution.from_json(
                ckpt.load_data(ckpt.solution_name.format(solution_id)))
        except Exception:
            if self.verbose:
                self.logger.warning(f'Solution id: {solution_id} not found!')
            return None

    def save_state(self, data={}):
        '''Saves the current population of the evolution.

        A file named data_X.ckpt (by default) is created after each generation, 
        where X is the number of evaluations, including the following information:
        - evals
        - current population'''

        data['evals'] = self.evals
        data['population'] = [s.to_json() for s in self.population]

        # creates the data checkpoint
        file_name = ckpt.data_name.format(self.evals)
        saved = ckpt.save_data(data, file_name)

        # remove solution files already evaluated if data ckpt exists
        if saved: 
            for i in range(self.evals):
                ckpt.delete_data(ckpt.solution_name.format(i))
            self.logger.debug(f'Checkpoint [{file_name}] created.')

    def load_state(self):
        '''Loads the last generation saved as checkpoint.
        Seaches for the most recent data_X.ckpt file, where X is the number of evaluations.'''

        # searches for data checkpoints
        data_ckpts = ckpt.get_files_with_name(ckpt.data_name.format('*'))

        if len(data_ckpts) == 0:
            self.logger.debug('No checkpoint found.')
            return None

        last_ckpt = max(data_ckpts, key=lambda c: ckpt.natural_key(c))
        data = ckpt.load_data(last_ckpt)

        self.evals = data['evals']
        self.population = [
            Solution.from_json(s) for s in data['population']
        ]

        if self.verbose:
            self.logger.debug(f'Latest checkpoint file found: {last_ckpt}')
            self.logger.debug(f'Current evals: {self.evals}/{self.max_evals}')
            self.logger.debug(f'Population size: {len(self.population)}')

        return data