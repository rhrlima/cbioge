import copy

class Solution():
    '''Represents a solution and common components that can be used in a
    wide range of problems. Custom data should use the data dictionary to store
    statistics or other useful info accessed by the problem. 
    
    The search engine will use the basic components, and the problem
    (usually a custom class) will make use of most of it, or even more.'''

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
        if not isinstance(other, Solution):
            return False
        return self.to_json() == other.to_json()

    def to_json(self):
        return self.__dict__

    def copy(self, deep=False):
        if deep: return copy.deepcopy(self)
        return Solution(copy.deepcopy(self.genotype))

    @classmethod
    def from_json(cls, json_data):
        if type(json_data) is not dict:
            return cls()
        return cls(**json_data)
