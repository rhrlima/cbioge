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

    id = None
    phenotype = None
    time = None
    params = None

    def copy(self, deep=False):
        solution = GESolution(self.genotype[:])
        if deep:
            solution.id = self.id
            solution.fitness = self.fitness
            solution.phenotype = self.phenotype
            solution.evaluated = self.evaluated
            solution.time = self.time
            solution.params = self.params
        return solution

    def to_json(self):
        json_solution = {
            'id': self.id,
            'genotype': self.genotype,
            'phenotype': self.phenotype,
            'fitness': self.fitness,
            'evaluated': self.evaluated,
            'time': self.time,
            'params': self.params,
        }
        return json_solution

    def from_json(json_solution):
        self.id = json_solution['id']
        self.genotype = json_solution['genotype']
        self.phenotype = json_solution['phenotype']
        self.fitness = json_solution['fitness']
        self.evaluated = json_solution['evaluated']
        self.time = json_solution['time']
        self.params = json_solution['params']
