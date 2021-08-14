from ..algorithms import GESolution
from ..algorithms import BaseEvolutionaryAlgorithm
from ..utils import checkpoint as ckpt


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

    def __init__(self, problem, 
        seed=None, 
        pop_size=10, 
        max_evals=20, 
        verbose=False, 
        selection=None, 
        replacement=None, 
        crossover=None, 
        mutation=None):

        super().__init__(problem, seed, pop_size, max_evals, verbose, selection, 
            replacement, crossover, mutation)

        self.unique_solutions = list()

    def create_population(self, size):
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

    def evaluate_solution(self, solution: GESolution):

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
            log_text = f'Solution {solution.id} fit: {float(solution.fitness):.2f} gen: {solution}'
            self.logger.debug(log_text)

    def evaluate_population(self, population):
        for s in population:
            self.evaluate_solution(s)

    def accept_solution(self, solution):
        # maintain only unique solutions
        if solution is None or solution.genotype in self.unique_solutions:
           return False
        self.unique_solutions.append(solution.genotype[:])
        return True

    def execute(self, checkpoint=False):

        if checkpoint: self.load_state()

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
                if offspring is None:
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
    
    def save_state(self):
        '''Saves the current population of the evolution'''

        data = {
            'evals': self.evals,
            'population': [s.to_json() for s in self.population],
            'unique': self.unique_solutions,
        }

        # creates the data checkpoint
        file_name = ckpt.data_name.format(self.evals)
        saved = ckpt.save_data(data, file_name)

        # remove solution files already evaluated if data ckpt exists
        if saved: 
            for i in range(self.evals):
                ckpt.delete_data(ckpt.solution_name.format(i))
            self.logger.debug(f'Checkpoint [{file_name}] created.')

    def load_state(self):
        '''Tries to continue the evolution from the last checkpoint'''

        # searches for data checkpoints
        data_ckpts = ckpt.get_files_with_name(ckpt.data_name.format('*'))

        if len(data_ckpts) == 0:
            self.logger.debug('No checkpoint found.')
            return

        last_ckpt = max(data_ckpts, key=lambda c: ckpt.natural_key(c))
        data = ckpt.load_data(last_ckpt)

        self.evals = data['evals']
        self.population = [
            GESolution.from_json(s) for s in data['population']
        ]

        if 'unique' in data: self.unique_solutions = data['unique']

        self.logger.debug(f'Latest checkpoint file found: {last_ckpt}')
        self.logger.debug(f'Current evals: {self.evals}/{self.max_evals}')
        self.logger.debug(f'Population size: {len(self.population)}')
        self.logger.debug(f'Unique solutions: {len(self.unique_solutions)}')