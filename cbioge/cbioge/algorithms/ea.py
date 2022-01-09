from __future__ import annotations
from typing import List, TYPE_CHECKING
import logging

import numpy as np

from .solution import Solution
from .selection import TournamentSelection
from .replacement import ReplaceWorst
from ..utils import checkpoint as ckpt

if TYPE_CHECKING:
    from .operators import (
        CrossoverOperator,
        MutationOperator,
        ReplacementOperator,
        SelectionOperator
    )
    from ..problems import BaseProblem


class BaseEvolutionaryAlgorithm:
    '''Base structure for the evolutionary search.

    It defines basic operations for creating solutions, applying operators,
    logging, and saving the state of the algorithm.'''

    def __init__(self, problem: BaseProblem,
        pop_size: int=5,
        max_evals: int=10,
        verbose: bool=False,
        selection: SelectionOperator=TournamentSelection(2, 2, maximize=True),
        replacement: ReplacementOperator=ReplaceWorst(maximize=True),
        crossover: CrossoverOperator=None,
        mutation: MutationOperator=None,
        seed: int=None,
    ):

        self.problem = problem

        self.seed = seed
        self.pop_size = pop_size
        self.max_evals = max_evals

        self.selection = selection
        self.replacement = replacement
        self.crossover = crossover
        self.mutation = mutation

        self.verbose = verbose

        self.evals: int = 0
        self.population: list = []

        np.random.seed(seed=self.seed)
        self.logger = logging.getLogger('cbioge')

    def create_solution(self) -> Solution:
        '''Creates a solution following the encoding defined by the grammar-system
        in DSGE.

        Ex: [[a, b, c], [d, e, f], ...] where a-f represents integer values.'''
        return Solution(self.problem.parser.create_solution())

    def evaluate_solution(self, solution: Solution) -> None:
        raise NotImplementedError('Not implemented yet.')

    def apply_selection(self) -> List[Solution]:
        return self.selection.execute(self.population)

    def apply_crossover(self, parents: List[Solution]) -> Solution:
        if self.crossover is None:
            return parents[0].copy()
        return self.crossover.execute(parents)

    def apply_mutation(self, offspring: Solution) -> Solution:
        if self.mutation is None:
            return offspring.copy()
        return self.mutation.execute(offspring)

    def apply_replacement(self, offspring_pop: List[Solution]) -> List[Solution]:
        return self.replacement.execute(self.population, offspring_pop)

    def execute(self, checkpoint: bool=False) -> Solution:
        raise NotImplementedError('Not implemented yet.')

    def print_progress(self) -> None:
        best = max(self.population, key=lambda x: x.fitness)
        log_text = (
            f'evals: {self.evals}/{self.max_evals} ' +
            f'best so far: {float(best.fitness): .2f} gen: {best.genotype}'
        )
        self.logger.info(log_text)

    def save_solution(self, solution: Solution) -> None:
        json_solution = solution.to_json()
        filename = ckpt.SOLUTION_NAME.format(solution.id)
        ckpt.save_data(json_solution, filename)

    def load_solution(self, solution_id: int) -> Solution:
        try:
            return Solution.from_json(
                ckpt.load_data(ckpt.SOLUTION_NAME.format(solution_id)))
        except FileNotFoundError:
            if self.verbose:
                warn_text = f'Solution id: {solution_id} not found!'
                self.logger.warning(warn_text)
            return None

    def save_state(self, data: dict=None) -> None:
        '''Saves the current population of the evolution.

        A file named data_X.ckpt (by default) is created after each generation,
        where X is the number of evaluations, including the following information:
        - evals
        - current population'''

        if data is None:
            data = dict()

        data['evals'] = self.evals
        data['population'] = [s.to_json() for s in self.population]

        # creates the data checkpoint
        file_name = ckpt.DATA_NAME.format(self.evals)
        saved = ckpt.save_data(data, file_name)

        # remove solution files already evaluated if data ckpt exists
        if saved:
            for i in range(self.evals):
                ckpt.delete_data(ckpt.SOLUTION_NAME.format(i))
            if self.verbose:
                debug_text = f'Checkpoint [{file_name}] created.'
                self.logger.debug(debug_text)

    def load_state(self) -> dict:
        '''Loads the last generation saved as checkpoint.
        Seaches for the most recent data_X.ckpt file, where X is the number of evaluations.'''

        # searches for data checkpoints
        data_ckpts = ckpt.get_files_with_name(ckpt.DATA_NAME.format('*'))

        if len(data_ckpts) == 0:
            if self.verbose:
                self.logger.debug('No checkpoint found.')
            return None

        last_ckpt = max(data_ckpts, key=ckpt.natural_key)
        data = ckpt.load_data(last_ckpt)

        self.evals = data['evals']
        self.population = [
            Solution.from_json(s) for s in data['population']
        ]

        if self.verbose:
            self.logger.debug('Latest checkpoint file found: %s', last_ckpt)
            self.logger.debug('Current evals: %d/%d', self.evals, self.max_evals)
            self.logger.debug('Population size: %d', len(self.population))

        return data
