from .problem import BaseProblem
import numpy as np


class SymbolicRegressionProblem(BaseProblem):

    def __init__(self, parser_, eq=None):
        self.parser = parser_
        self.equation = eq
        self.inputs = np.arange(-1, 1, 0.1)
        self.known_best = None

    def map_genotype_to_phenotype(self, genotype):
        deriv = self.parser.parse(genotype)
        if not deriv:
            return None
        return ''.join(deriv)

    def evaluate(self, solution):

        solution.phenotype = self.map_genotype_to_phenotype(solution.genotype)

        if not solution.phenotype:
            return float('inf'), None

        try:
            eq1 = lambda x: eval(solution.phenotype)
            diff_func = lambda x: (eq1(x) - self.equation(x))**2
            diff = sum(map(diff_func, self.inputs))
        except Exception:
            return float('inf'), None

        return diff, solution.phenotype


def inv(x):
    return 1 / x
