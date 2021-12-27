from ..algorithms import Solution
from .dsge import GrammaticalEvolution


class RandomGrammaticalEvolution(GrammaticalEvolution):
    '''Random search based on the DSGE algorithm.

    Follows the same base structure of GA algorithm, without applying genetic
    operators, and seaching the best solutions generated randomly.'''

    def __init__(self, problem,
        seed=None,
        pop_size=10,
        max_evals=20,
        verbose=False
    ):

        super().__init__(problem, seed, pop_size, max_evals, verbose)

    def execute(self, checkpoint: bool=False) -> Solution:

        self.evals = 0
        self.population = []
        self.unique_solutions = []

        if checkpoint:
            self.load_state()

        self.print_progress()

        offspring_pop = []
        while self.evals < self.max_evals:

            # creates a new population from recombining the current one
            index = 0
            while len(offspring_pop) < self.pop_size:
                solution = self.create_solution()
                solution.id = self.evals + index
                offspring_pop.append(solution)
                index += 1

            self.evaluate_population(offspring_pop)

            self.population = self.apply_replacement(offspring_pop)

            self.evals += self.pop_size
            offspring_pop.clear()

            self.save_state()
            self.print_progress()

        return max(self.population, key=lambda x: x.fitness)
