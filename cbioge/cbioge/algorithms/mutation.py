import numpy as np

from .solution import Solution
from .operators import MutationOperator
from ..grammars import Grammar


class PointMutation(MutationOperator):
    '''Selects a random value present in the solution and replaces it by a new
    valid value according to the grammar.

    # Parameters
    mut_rate: chance to apply the operator
    parser: parser object needed to replace the values'''

    def __init__(self, parser: Grammar,
        mut_rate: float,
        start_index: int=0,
        end_index: int=None
    ):
        super().__init__(mut_rate)

        self.parser = parser

        self.start_index = start_index
        self.end_index = end_index

    def __str__(self):

        return 'Point Mutation'

    def execute(self, solution: Solution) -> Solution:

        offspring = solution.copy()
        if np.random.rand() > self.mut_rate:
            return offspring

        if self.end_index is None:
            self.end_index = len(solution.genotype)

        # get one random block from the solution
        genes_len = 0
        while genes_len == 0:
            block_idx = np.random.randint(self.start_index, self.end_index)
            genes_len = len(offspring.genotype[block_idx])

        # get one random value from the block
        gene_idx = np.random.randint(0, genes_len)

        # symbol on the gene index
        symb = self.parser.nonterm[block_idx]

        # max options for the symb
        max_value = len(self.parser.rules[symb])

        if max_value > 1:
            curr_value = offspring.genotype[block_idx][gene_idx]
            new_value = curr_value
            while new_value == curr_value:
                new_value = np.random.randint(0, max_value)
            offspring.genotype[block_idx][gene_idx] = new_value

        return offspring


class TerminalMutation(PointMutation):
    '''Follows similar behavior than the Point Mutation.
    Changes a value for a new valid one, starting from a given index

    Ex: [[1,2,3], | [4,5,6], [7,8,9]] start_index = 3\n
    means that values [1,2,3] will not be changed.'''

    def __init__(self, parser: Grammar,
        mut_rate: float,
        start_index: int=0
    ):
        super().__init__(parser, mut_rate, start_index=start_index)

    def __str__(self):

        return 'Terminal Point Mutation'


class NonterminalMutation(PointMutation):
    '''Follows similar behavior than the Point Mutation.
    Changes a value for a new valid one, up to a given index

    Ex: [[1,2,3], [4,5,6], | [7,8,9]] start_index = 6\n
    means that values [7,8,9] will not be changed.'''

    def __init__(self, parser: Grammar,
        mut_rate: float,
        end_index: int=None
    ):
        super().__init__(parser, mut_rate, end_index=end_index)

    def __str__(self):

        return 'Nonterminal Point Mutation'
