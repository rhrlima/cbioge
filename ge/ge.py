import numpy as np
import random

rand = np.random

POP_SIZE = 5
MIN_GENES = 1
MAX_GENES = 10
MAX_EVALS = 2000

CROSS_RATE = 0.8
MUT_RATE = 0.1

MINIMIZE = False

problem = None
grammar = None


class Solution:

	genotype = None
	phenotype = None
	fitness = None
	evaluated = False

	def __init__(self, genes):
		self.genotype = genes

	def copy(self):
		return Solution(self.genotype[:])

	def __str__(self):
		return str(self.genotype)


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
	if not solution.evaluated:
		solution.fitness = problem.evaluate(solution, 1)
		solution.evaluated = True


def evaluate_population(population):
	for solution in population:
		evaluate_solution(solution)


def selection(population):
	p1 = None
	p2 = None
	p1 = random.choice(population)
	while not p2 or p1 is p2:
		p2 = random.choice(population)
	return [p1, p2]


def crossover(parents):
	off1 = parents[0].copy()
	off2 = parents[1].copy()
	if rand.rand() < CROSS_RATE:
		p1 = off1.genotype[:]
		p2 = off2.genotype[:]
		min_ = min(len(p1), len(p2))
		cut = rand.randint(0, min_)
		off1.genotype = np.concatenate((p1[:cut], p2[cut:]))
		off2.genotype = np.concatenate((p2[:cut], p1[cut:]))
	return [off1, off2]


def mutate(offspring):
	if rand.rand() < MUT_RATE:
		for off in offspring:
			index = rand.randint(0, len(off.genotype))
			off.genotype[index] = rand.randint(0, 255)


def prune(offspring):
	# if rand.rand() < PRUNE_RATE:
	for off in offspring:
		cut = rand.randint(1, len(off.genotype))
		off.genotype = of.genotype[:cut]


def duplicate(offspring):
	pass


def replace(population, offspring):
	population += offspring
	population.sort(key=lambda x: x.fitness, reverse=not MINIMIZE)
	population.pop()
	population.pop()
	for p in population:
		print(p.fitness)


def execute():
	
	population = create_population(POP_SIZE)
	evaluate_population(population)

	evals = len(population)

	while evals < MAX_EVALS:
		
		print('evals: {:4}/{:4}'.format(evals, MAX_EVALS))

		parents = selection(population)

		offspring = crossover(parents)
		
		mutate(offspring)

		evaluate_population(offspring)

		replace(offspring, population)

		evals += len(offspring)

	population.sort(key=lambda x:x.fitness, reverse=not MINIMIZE)
	return population[0]
