import copy

class GESolution():

    def __init__(self, gen=[], json_data=None):

        self.id = None
        self.genotype = gen
        self.phenotype = None
        self.fitness = -1
        self.evaluated = False
        self.time = None
        self.params = None

        if json_data is not None:
            self.initialize_from_json(json_data)

    def __str__(self):
        return str(self.genotype)

    def to_json(self):
        return self.__dict__

    def initialize_from_json(self, json_data):
        for key in self.__dict__:
            self.__dict__[key] = json_data[key]

    def copy(self):
        new_solution = copy.deepcopy(self)
        new_solution.evaluated = False
        return new_solution

if __name__ == '__main__':
    
    # create from list
    s = GESolution([[0,0],[0,0,0],[0]])
    print(s)

    # create from json
    json_string = {'id': None, 'genotype': [[1, 1], [1, 1, 1], [1]], 'phenotype': None, 'fitness': -1, 'evaluated': False, 'time': None, 'params': None}
    s = GESolution(json_data=json_string)
    print(s)

    # export json
    json_string = s.to_json()
    print(json_string)

    # copy and compare
    s_copy = s.copy()
    s_copy2 = GESolution(json_data=json_string)
    print(s is s_copy, s == s_copy)
    print(s is s_copy2, s == s_copy2)