from __future__ import annotations
from typing import List, TYPE_CHECKING

from ..algorithms import BaseEvolutionaryAlgorithm

# TODO study better way to handle this
# avoids import cycles while using typing
if TYPE_CHECKING:
    from .operators import (
        SelectionOperator,
        ReplacementOperator,
        CrossoverOperator,
        MutationOperator,
    )
    from ..algorithms import Solution
    from ..problems import BaseProblem


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):
    '''Genetic Algorithm modified to work with the DSGE encoding.

    This modified version mainstains a list of unique solutions stored, which
    helps increasing the diversity.'''

    def __init__(self, problem: BaseProblem,
        pop_size: int=10,
        max_evals: int=20,
        verbose: bool=False,
        selection: SelectionOperator=None,
        replacement: ReplacementOperator=None,
        crossover: CrossoverOperator=None,
        mutation: MutationOperator=None,
        seed: int=None
    ):

        super().__init__(problem, pop_size, max_evals, verbose, selection,
            replacement, crossover, mutation, seed)

        self.unique_solutions = []

    def create_population(self, size: int) -> List[Solution]:
        population = []
        index = 0
        while len(population) < size:
            solution = self.create_solution()
            if self.accept_solution(solution):
                solution.id = index
                population.append(solution)
                self.save_solution(solution)
                index += 1
        return population

    def evaluate_solution(self, solution: Solution) -> None:
        '''Evaluates a solution

        This procedure is ignored if the solution has been evaluated already.
        It calls the defined mapping method followed by the evaluate method,
        both defined in the problem assigned.'''

        # skip solutions already executed
        if solution.evaluated:
            if self.verbose:
                log_text = f'Solution {solution.id} already evaluated. Skipping...'
                self.logger.debug(log_text)
            return

        # performs mapping and evaluates taking the time spent
        self.problem.map_genotype_to_phenotype(solution)
        self.problem.evaluate(solution)

        solution.evaluated = True

        # updates the solution file
        self.save_solution(solution)

        if self.verbose:
            log_text = (f'Solution {solution.id} '
                + f'fit: {float(solution.fitness):.2f} gen: {solution}')
            self.logger.debug(log_text)


    def evaluate_population(self, population: List[Solution]) -> None:
        for solution in population:
            self.evaluate_solution(solution)

    def accept_solution(self, solution: Solution) -> bool:
        # maintain only unique solutions
        if solution is None or solution.genotype in self.unique_solutions:
            return False
        self.unique_solutions.append(solution.genotype[:])
        return True

    def execute(self, checkpoint: bool=False) -> Solution:
        '''Runs the evolution.

        The parameter checkpoint will define if the execution will be from scratch
        or continue from a previous checkpoint (if any).'''

        if checkpoint:
            self.load_state()

        #Initial Evaluation
        if len(self.population) == 0:
            self.population = self.create_population(self.pop_size)
            self.evaluate_population(self.population)
            self.evals = len(self.population)
            self.save_state()

        self.print_progress()

        offspring_pop = []
        while self.evals < self.max_evals:

            # creates a new population from recombining the current one
            index = 0
            while len(offspring_pop) < self.pop_size:

                # tries to load solution if starting from checkpoint
                offspring = self.load_solution(self.evals + index)

                # creates new solution if load fails
                # TODO temp fix for possible infinite loop when loading 
                # solution that already exists
                # rework the load solution strategy
                if offspring is None or not self.accept_solution(offspring):
                    # apply selection and recombination operators
                    parents = self.apply_selection()
                    offspring = self.apply_crossover(parents)
                    offspring = self.apply_mutation(offspring)
                    offspring.id = self.evals + index # check

                if self.accept_solution(offspring):
                    self.save_solution(offspring)
                    offspring_pop.append(offspring)
                    index += 1

            self.evaluate_population(offspring_pop)

            self.population = self.apply_replacement(offspring_pop)

            self.evals += self.pop_size
            offspring_pop.clear()

            self.save_state()
            self.print_progress()

        return max(self.population, key=lambda x: x.fitness)

    def save_state(self, data: dict=None) -> None:
        '''Saves the current population and evaluations by default.
        Additionally saves the list of unique solutions'''

        data = {
            'unique': self.unique_solutions,
        }

        # super method will add population and evals
        super().save_state(data)

    def load_state(self) -> dict:

        data = super().load_state()

        if data is None:
            return

        # super method already loads population and evals
        if 'unique' in data:
            self.unique_solutions = data['unique']

        if self.verbose:
            debug_text = f'Unique solutions: {len(self.unique_solutions)}'
            self.logger.debug(debug_text)
