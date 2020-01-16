class GESolution():

    def __init__(self, gen=[], json_data=None):

        self.id = None
        self.genotype = gen
        self.phenotype = None
        self.fitness = None
        self.evaluated = False
        self.time = None
        self.params = None

        if not json_data is None:
            self.initialize_from_json(json_data)

    def __str__(self):
        return str(self.genotype)

    def to_json(self):
        return self.__dict__

    def initialize_from_json(self, json_data):
        for key in self.__dict__:
            self.__dict__[key] = json_data[key]
        # self.id = json_data['id']
        # self.genotype = json_data['genotype']
        # self.phenotype = json_data['phenotype']
        # self.fitness = json_data['fitness']
        # self.evaluated = json_data['evaluated']
        # self.time = json_data['time']
        # self.params = json_data['params']
