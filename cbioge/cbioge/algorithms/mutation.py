import numpy as np

from .operators import GeneticOperator


class PointMutation(GeneticOperator):

    ''' Point Mutation changes a list of solutions by selecting a random
        point and generating a new value for that position (repeate for
        each solution)

        mut_rate: chance to apply the operator
        min_value: min possible value for the solution
        max_value: max possible value for the solution
    '''

    def __init__(self, mut_rate, min_value=0, max_value=1):
        self.mut_rate = mut_rate
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self):
        return 'Point Mutation'

    def execute(self, offspring):
        if np.random.rand() < self.mut_rate:
            for off in offspring:
                index = np.random.randint(0, len(off.genotype))
                off.genotype[index] = np.random.randint(
                    self.min_value, self.max_value)


class DSGEMutation(GeneticOperator):

    '''Changes one value for a new valid one

        mut_rate: chance to apply the operator
        parser: parser object needed to replace the values
    '''

    def __init__(self, mut_rate, parser):
        self.mut_rate = mut_rate
        self.parser = parser

        self.start_index = 0
        self.end_index = None

    def __str__(self):

        return 'DSGE Point Mutation'

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


class DSGETerminalMutation(DSGEMutation):

    '''Changes a value for a new valid one, starting from a given index'''

    def __init__(self, parser, start_index, mut_rate=1.0):
        super().__init__(mut_rate, parser)
        
        self.start_index = start_index

    def __str__(self):

        return 'DSGE Terminal Mutation'


class DSGENonterminalMutation(DSGEMutation):

    '''Changes a value for a new valid one, restricted to and index'''

    def __init__(self, parser, end_index, mut_rate=1.0):
        super().__init__(mut_rate, parser)

        self.end_index = end_index

    def __str__(self):

        return 'DSGE Nonterminal Mutation'

class DSGENonTerminalMutation(DSGEMutation):

    pass