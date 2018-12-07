
class Solution:

	genotype = None
	phenotype = None
	fitness = -1#None
	
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