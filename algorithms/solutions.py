class BaseSolution:

	genes = None
	fitness = None
	data = {}
	evaluated = False

	def __init__(self, genes, fitness=None, data={}):
		self.genes = genes
		self.fitness = fitness
		self.data = data
		#self.evaluated = False

	def copy(self, deep=False):
		solution = BaseSolution(self.genes[:])
		if deep:
			solution.fitness = self.fitness
			solution.data = self.data
			solution.evaluated = self.evaluated
		return solution

	def __str__(self):
		return str(self.genes)


class GESolution:

	genotype = None
	phenotype = None
	fitness = -1
	evaluated = False

	def __init__(self, genotype):
		self.genotype = genotype

	def copy(self, deep=False):
		solution = GESolution(self.genotype[:])
		if deep:
			solution.fitness = self.fitness
			solution.phenotype = self.phenotype
			solution.evaluated = self.evaluated
		return solution

	def __str__(self):
		return str(self.genotype)