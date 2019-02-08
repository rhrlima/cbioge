class BaseSolution:

	genes = None
	fitness = None

	def __init__(self, genes, fitness=None):
		self.genes = genes
		self.fitness = fitness

	def __str__(self):
		return str(self.genes)


class GeneticSolution:

	genotype = None
	phenotype = None
	fitness = None
	data = {}
	evaluated = False

	def __init__(self, genes, phen=None):
		self.genotype = genes

	def copy(self, deep=False): #shallow
		solution = Solution(self.genotype[:])
		if deep:
			solution.phenotype = self.phenotype
			solution.fitness = self.fitness
			solution.evaluated = self.evaluated
		return solution

	def __str__(self):
		return str(self.genotype)