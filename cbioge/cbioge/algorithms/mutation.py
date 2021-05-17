import numpy as np

from cbiogeevolution.algorithms.operators import GeneticOperator


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

    def __init__(self, mut_rate, parser):
        self.mut_rate = mut_rate
        self.parser = parser

    def __str__(self):

        return 'DSGE Point Mutation'

    def execute(self, solution):
        for gidx, genes in enumerate(solution.genotype):
            symb = self.parser.NT[gidx] # symbol on the gene index
            max_value = len(self.parser.GRAMMAR[symb]) # options for the symb
            for i, _ in enumerate(genes):
                if np.random.rand() < self.mut_rate:
                    print('[operator] mutation applied to gene', i)
                    new_val = genes[i]
                    while new_val == genes[i] and max_value > 1:
                        new_val = np.random.randint(0, max_value)
                    # if max_value == 1:
                    #     print('only one option', genes[i])
                    # print('MUTOU', _, 'to', new_val, 'out of', max_value)
                    genes[i] = new_val


class DSGETerminalMutation(GeneticOperator):

    '''Changes a value for a new valid one, starting from a given index
    '''

    def __init__(self, mut_rate, parser, start_index):
        self.mut_rate = mut_rate
        self.parser = parser
        self.start_index = start_index

    def __str__(self):

        return 'DSGE Restricted Point Mutation'

    def execute(self, solution):
        for gidx, genes in enumerate(solution.genotype):

            if gidx < self.start_index:
                continue

            symb = self.parser.NT[gidx] # symbol on the gene index
            max_value = len(self.parser.GRAMMAR[symb]) # options for the symb
            #print(symb, max_value)
            for i, _ in enumerate(genes):
                if np.random.rand() < self.mut_rate:
                    new_val = genes[i]
                    while new_val == genes[i] and max_value > 1:
                        new_val = np.random.randint(0, max_value)
                    # if max_value == 1:
                    #     print('only one option', genes[i])
                    #print('MUTOU', _, 'to', new_val, 'out of', max_value)
                    genes[i] = new_val


class DSGENonTerminalMutation(GeneticOperator):

    '''Changes a value for a new valid one, only for non-terminals
    '''

    def __init__(self, mut_rate, parser, end_index):
        self.mut_rate = mut_rate
        self.parser = parser
        self.end_index = end_index

    def __str__(self):

        return 'DSGE Restricted Point Mutation'

    def execute(self, solution):
        for gidx, genes in enumerate(solution.genotype):

            if gidx >= self.end_index:
                continue

            symb = self.parser.NT[gidx] # symbol on the gene index
            max_value = len(self.parser.GRAMMAR[symb]) # options for the symb
            #print(symb, max_value)
            for i, _ in enumerate(genes):
                if np.random.rand() < self.mut_rate:
                    new_val = genes[i]
                    while new_val == genes[i] and max_value > 1:
                        new_val = np.random.randint(0, max_value)
                    # if max_value == 1:
                    #     print('only one option', genes[i])
                    #print('MUTOU', _, 'to', new_val, 'out of', max_value)
                    genes[i] = new_val
