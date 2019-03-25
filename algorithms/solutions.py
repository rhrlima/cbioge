class BaseSolution:

    genotype = None
    fitness = None
    data = {}
    evaluated = False

    def __init__(self, genotype):
        self.genotype = genotype

    def copy(self, deep=False):

        solution = BaseSolution(self.genotype[:])
        if deep:
            solution.fitness = self.fitness
            solution.data = self.data
            solution.evaluated = self.evaluated
        return solution

    def __str__(self):
        return str(self.genotype)


class GESolution(BaseSolution):

    phenotype = None

    def copy(self, deep=False):
        solution = GESolution(self.genotype[:])
        if deep:
            solution.fitness = self.fitness
            solution.phenotype = self.phenotype
            solution.evaluated = self.evaluated
        return solution
