from .problem import BaseProblem


class StringMatchProblem(BaseProblem):

    def __init__(self, parser_, target='Hello World!'):
        self.parser = parser_
        self.target = target

    def map_genotype_to_phenotype(self, genotype):
        deriv = self.parser.parse(genotype)
        if not deriv:
            return None
        return ''.join(deriv)

    def evaluate(self, solution):

        guess = self.map_genotype_to_phenotype(solution.genotype)

        if not guess:
            return float('inf'), None

        guess = guess.replace('\"', '')
        fitness = max(len(self.target), len(guess))
        # Loops as long as the shorter of two strings
        for (t_p, g_p) in zip(self.target, guess):
            if t_p == g_p:
                # Perfect match.
                fitness -= 1
            else:
                # Imperfect match, find ASCII distance to match.
                fitness -= 1 / (1 + (abs(ord(t_p) - ord(g_p))))
        return fitness, guess
