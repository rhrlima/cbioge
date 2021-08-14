import copy

class GESolution():

    def __init__(self, 
        genotype=[], 
        phenotype=None, 
        fitness=-1, 
        evaluated=False, 
        data={}, 
        id=None):

        self.id = id
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.evaluated = evaluated
        self.data = data

    def __str__(self):
        return str(self.genotype)

    def __eq__(self, other):
        if not isinstance(other, GESolution):
            return False
        return self.to_json() == other.to_json()

    def to_json(self):
        return self.__dict__

    def copy(self, deep=False):
        if deep: return copy.deepcopy(self)
        return GESolution(copy.deepcopy(self.genotype))

    @classmethod
    def from_json(cls, json_data):
        if type(json_data) is not dict:
            return cls()
        return cls(**json_data)
