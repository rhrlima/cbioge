import copy

class GESolution():

    def __init__(self, gen=[], json_data=None):
        self.id = None
        self.genotype = gen
        self.phenotype = None
        self.fitness = -1
        self.evaluated = False
        #self.time = None
        #self.params = None
        self.data = {}

        if json_data is not None:
            self.initialize_from_json(json_data)

    def __str__(self):
        return str(self.genotype)

    def __eq__(self, other):
        if not isinstance(other, GESolution):
            return False
        return self.to_json() == other.to_json()

    # def __hash__(self):
    #     return hash(self.genotype)

    def to_json(self):
        return self.__dict__

    def initialize_from_json(self, json_data):
        if type(json_data) is not dict:
            return

        for key in self.__dict__:
            if key in json_data:
                self.__dict__[key] = json_data[key]

    def copy(self, deep=False):
        if deep: return copy.deepcopy(self)
        return GESolution(copy.deepcopy(self.genotype))
