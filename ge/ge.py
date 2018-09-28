import numpy as np
import random

rand = np.random

POP_SIZE = 5
MIN_GENES = 1
MAX_GENES = 10
MAX_EVALS = 2000

problem = None
grammar = None


class Solution:

	genotype = None
	phenotype = None
	fitness = None
	evaluated = False

	def __init__(self, genes):
		self.genotype = genes


def create_solution(min_, max_=None):
	if not max_:
		max_ = min_
		min_ = 0
	genes = rand.randint(0, 255, rand.randint(min_, max_))
	return Solution(genes)


def create_population(size):
	population = []
	for _ in range(size):
		population.append(create_solution(MIN_GENES, MAX_GENES))
	return population


def evaluate_solution(solution):
	solution.fitness = -1
	solution.evaluated = True
	print(solution.genotype)


def evaluate_population(population):
	for solution in population:
		evaluate_solution(solution)


def selection(population):
	p1 = None
	p2 = None
	
	while not parents or parents[0] == parents[1]:
		if parents != None:
			print(parents[0].genotype, parents[1].genotype, parents[0]==parents[1])
		else:
			print('none')
		parents = random.choices(population=population, k=2)
	return parents


def crossover(parents):
	pass


def mutate(offspring):
	pass


def prune(offspring):
	pass


def duplicate(offspring):
	pass


def execute():
	
	print('population')
	population = create_population(POP_SIZE)
	evaluate_population(population)

	evals = len(population)

	#while evals < MAX_EVALS:
	print('parents')
	parents = selection(population)

	offspring = crossover(parents)
	mutate(offspring)

	#evals += len(offspring)