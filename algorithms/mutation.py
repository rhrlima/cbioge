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

        self.start_index = None
        self.end_index = None

    def __str__(self):

        return 'DSGE Point Mutation'

    def execute(self, solution):
        if self.start_index is None:
            self.start_index = 0
        if self.end_index is None:
            self.end_index = len(solution.genotype)

        solution_copy = solution.copy()

        block_idx = np.random.randint(self.start_index, self.end_index)

        genes_len = len(solution.genotype[block_idx])
        while genes_len == 0:
            #print('not enough genes')
            block_idx = np.random.randint(self.start_index, self.end_index)
            genes_len = len(solution.genotype[block_idx])

        gene_idx = np.random.randint(0, genes_len)

        if np.random.rand() < self.mut_rate:
            symb = self.parser.NT[block_idx] # symbol on the gene index
            max_value = len(self.parser.GRAMMAR[symb]) # options for the symb
            #print('values', max_value)
            if max_value > 1:
                curr_value = solution.genotype[block_idx][gene_idx]
                new_value = solution.genotype[block_idx][gene_idx]
                while new_value == curr_value:
                    new_value = np.random.randint(0, max_value)
                solution_copy.genotype[block_idx][gene_idx] = new_value
                #print('mutou', curr_value, 'para', new_value)
        return solution_copy


class DSGETerminalMutation(DSGEMutation):

    '''Changes a value for a new valid one, starting from a given index'''

    def __init__(self, mut_rate, parser, start_index):
        super().__init__(mut_rate, parser)
        
        self.start_index = start_index

    def __str__(self):

        return 'DSGE Terminal Mutation'


class DSGENonterminalMutation(DSGEMutation):

    '''Changes a value for a new valid one, restricted to and index'''

    def __init__(self, mut_rate, parser, end_index):
        super().__init__(mut_rate, parser)

        self.end_index = end_index

    def __str__(self):

        return 'DSGE Nonterminal Mutation'
