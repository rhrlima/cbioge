import logging

import numpy as np

from .solution import Solution
from .replacement import ReplaceWorst
from .selection import TournamentSelection
from ..utils import checkpoint as ckpt


class BaseEvolutionaryAlgorithm:

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