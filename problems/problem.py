class BaseProblem:

    def _map_genotype_to_phenotype(self, solution):
        raise NotImplementedError('Not implemented yet.')

    def evaluate(self, phenotype):
        raise NotImplementedError('Not implemented yet.')