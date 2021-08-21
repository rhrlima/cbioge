import numpy as np

from .operators import MutationOperator


class PointMutation(MutationOperator):
    '''Changes one value for a new valid one

    mut_rate: chance to apply the operator
    parser: parser object needed to replace the values'''

    def __init__(self, parser, mut_rate=1.0, start_index=0, end_index=None):
        super().__init__(mut_rate)

        self.parser = parser

        self.start_index = start_index
        self.end_index = end_index

    def __str__(self):

        return 'Point Mutation'

    def execute(self, solution):

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
    '''Changes a value for a new valid one, starting from a given index'''

    def __init__(self, parser, mut_rate=1.0, start_index=0):
        super().__init__(parser, mut_rate, start_index=start_index)

    def __str__(self):

        return 'Terminal Point Mutation'


class NonterminalMutation(PointMutation):
    '''Changes a value for a new valid one, up to a given index'''

    def __init__(self, parser, mut_rate=1.0, end_index=None):
        super().__init__(parser, mut_rate, end_index=end_index)

    def __str__(self):

        return 'Nonterminal Point Mutation'