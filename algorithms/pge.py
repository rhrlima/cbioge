#parallel GE modified to use qsub in evaluation
import os, sys
sys.path.append('..')

from multiprocessing import Pool, Manager
from utils import checkpoint

import numpy as np
import random
import time


rand = np.random

DEBUG = False
SEED = None

class Solution:

	genotype = None
	phenotype = None
	fitness = -1#None
	
	data = {}

	evaluated = False

	def __init__(self, genes):
		self.genotype = genes

	def copy(self): #shallow
		return Solution(self.genotype[:])

	def __str__(self):
		return str(self.genotype)


MAX_PROCESSES = 2

POP_SIZE = 5
MIN_GENES = 1
MAX_GENES = 10
MAX_EVALS = 100

CROSS_RATE = 0.8
MUT_RATE = 0.1
PRUN_RATE = 0.1
DUPL_RATE = 0.1

MINIMIZE = False

problem = None
grammar = None


def create_solution(min_, max_=None):
	if not max_:
		max_ = min_
		min_ = 0
	if min_ >= max_:
		raise ValueError('[create solution] min >= max')
	genes = rand.randint(0, 255, rand.randint(min_, max_))
	return Solution(genes)


def create_population(size):
	population = []
	for _ in range(size):
		population.append(create_solution(MIN_GENES, MAX_GENES))
	return population


def evaluate_solution(solution):
	if DEBUG: print('<{}> [evaluate] started: {}'.format(
		time.strftime('%x %X'), solution))

	if not solution.evaluated:
		if problem is None:
			if DEBUG:
				print('[evaluation] Problem is None, bypassing')
				solution.fitness = -1
			else:
				raise ValueError('Problem is None')
		else:
			fitness, model = problem.evaluate(solution)
	
	if DEBUG: print('<{}> [evaluate] ended: {}'.format(
		time.strftime('%x %X'), solution))
	
	return fitness, model


def evaluate_population(population):
	'''evaluate_population

		evaluates a population (list) of Solution objects using the 
		multiprocessing module

		it creates a pool of workers, each worker will evaluate one model
		and return the correspondent fitness (float and model (json string)

		at the end, the solution objet is updaded with the new information 
	'''
	pool = Pool(processes=MAX_PROCESSES)

	result = pool.map_async(evaluate_solution, population)
	
	pool.close()
	pool.join()

	for sol, res in zip(population, result.get()):
		fit, model = res
		sol.fitness = fit
		sol.phenotype = model
		sol.evaluated = True


def selection(population):
	if len(population) < 2:
		raise ValueError('[selection] population size is less than minimum (2)')
	p1 = None
	p2 = None
	p1 = random.choice(population)
	while not p2 or p1 is p2:
		p2 = random.choice(population)
	return [p1, p2]


def crossover(parents, prob):
	# testing, returing one child
	off1 = parents[0].copy()
	off2 = parents[1].copy()
	if rand.rand() < prob:
		p1 = off1.genotype[:]
		p2 = off2.genotype[:]
		min_ = min(len(p1), len(p2))
		cut = rand.randint(0, min_)
		off1.genotype = np.concatenate((p1[:cut], p2[cut:]))
		#off2.genotype = np.concatenate((p2[:cut], p1[cut:]))
	return [off1]#, off2]


def mutate(offspring, prob):
	if rand.rand() < prob:
		for off in offspring:
			index = rand.randint(0, len(off.genotype))
			off.genotype[index] = rand.randint(0, 255)


def prune(offspring, prob):
	if rand.rand() < prob:
		for off in offspring:
			if len(off.genotype) <= 1:
				if DEBUG: print('[prune] one gene, not applying:', off.genotype)
				continue
			cut = rand.randint(1, len(off.genotype))
			off.genotype = off.genotype[:cut]


def duplicate(offspring, prob):
	if rand.rand() < prob:
		for off in offspring:
			if len(off.genotype) > 1:
				cut = rand.randint(0, len(off.genotype))
			else:
				if DEBUG: print('[duplication] one gene, setting cut to 1:', off)
				cut = 1
			genes = off.genotype
			off.genotype = np.concatenate((genes, genes[:cut]))


def replace(population, offspring):
	
	population += offspring
	population.sort(key=lambda x: x.fitness, reverse=not MINIMIZE)

	for _ in range(len(offspring)):
		population.pop()


def execute():
	'''
	'''

	population = create_population(POP_SIZE)
	evaluate_population(population)

	population.sort(key=lambda x:x.fitness, reverse=not MINIMIZE)

	evals = len(population)

	if DEBUG:
		for i, p in enumerate(population):
			print(i, p.fitness, p)

	print('<{}> evals: {}/{} \tbest so far: {}\tfitness: {}'.format(
		time.strftime('%x %X'), 
		evals, MAX_EVALS, 
		population[0].genotype, 
		population[0].fitness)
	)

	save_state(evals, population)

	while evals < MAX_EVALS:

		parents = selection(population)

		offspring_pop = []

		for _ in population:

			offspring = crossover(parents, CROSS_RATE)
			
			mutate(offspring, MUT_RATE)

			prune(offspring, PRUN_RATE)

			duplicate(offspring, DUPL_RATE)

			offspring_pop += offspring

		evaluate_population(offspring_pop)

		replace(population, offspring_pop)

		evals += len(offspring_pop)

		if DEBUG:
			for i, p in enumerate(population):
				print(i, p.fitness, p)

		print('<{}> evals: {}/{} \tbest so far: {}\tfitness: {}'.format(
			time.strftime('%x %X'), 
			evals, MAX_EVALS, 
			population[0].genotype, 
			population[0].fitness)
		)
		
		save_state(evals, population)

	return population[0]


def save_state(evals, population):

	args = {
		'POP_SIZE': POP_SIZE, 
		'MIN_GENES': MIN_GENES, 
		'MAX_GENES': MAX_GENES, 
		'MAX_EVALS': MAX_EVALS, 

		'MAX_PROCESSES': MAX_PROCESSES, 

		'CROSS_RATE': CROSS_RATE, 
		'MUT_RATE': MUT_RATE, 
		'PRUN_RATE': PRUN_RATE, 
		'DUPL_RATE': DUPL_RATE, 

		'MINIMIZE': MINIMIZE, 

		'evals': evals
	}

	# files = ['pop.ckpt', 'args.ckpt']
	# for file in files:
	# 	if os.path.exists(file):
	# 		print('renaming "{0}" to "{0}.old"'.format(file))
	# 		os.rename(file, file+'.old')

	folder = 'checkpoints/'
	if not os.path.exists(folder): os.mkdir(folder)
	checkpoint.save_args(args, folder+'args_{}.ckpt'.format(evals))
	checkpoint.save_population(population, folder+'pop_{}.ckpt'.format(evals))

	# for file in files:
	# 	if os.path.exists(file):
	# 		print('removing "{0}.old" file'.format(file))
	# 		os.remove(file+'.old')