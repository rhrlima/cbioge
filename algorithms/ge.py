import os, sys
sys.path.append('..')

from multiprocessing import Pool, Manager
from utils import checkpoint

import glob
import numpy as np
import random
import re
import time

from .solutions import GESolution
from .ea import BaseEvolutionaryAlgorithm


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

	DEBUG = False

	def __init__(self, problem):
		super(GrammaticalEvolution, self).__init__(problem)

		self.SEED = None
		
		self.POP_SIZE = 5
		self.MAX_EVALS = 10

		self.MIN_GENES = 1
		self.MAX_GENES = 10
		self.MIN_VALUE = 0
		self.MAX_VALUE = 255

		self.selection = None
		self.crossover = None
		self.mutation = None
		self.prune = None
		self.duplication = None

		self.population = None
		self.evals = None


	def create_solution(self, min_size, max_size, min_value, max_value):
		if not max_size:
			max_size = min_size
			min_size = 0
		
		if min_size >= max_size:
			raise ValueError('[create solution] min >= max')

		genes = np.random.randint(min_value, max_value, np.random.randint(
			min_size, max_size))

		return GESolution(genes)


	def create_population(self, size):
		population = []
		for _ in range(size):
			solution = self.create_solution(self.MIN_GENES, self.MAX_GENES, 
					self.MIN_VALUE, self.MAX_VALUE)
			population.append(solution)
		return population


	def evaluate_solution(self, solution):
		if self.DEBUG: print('<{}> [evaluate] started: {}'.format(
			time.strftime('%x %X'), solution))

		if not solution.evaluated:
			if self.problem is None:
				if self.DEBUG:
					print('[evaluation] Problem is None, bypassing')
					solution.fitness = -1
				else:
					raise ValueError('Problem is None')
			else:
				fitness, model = self.problem.evaluate(solution)
		
		if self.DEBUG: print('<{}> [evaluate] ended: {}'.format(
			time.strftime('%x %X'), solution))
		
		return fitness, model


	def evaluate_population(self, population):

		pool = Pool(processes=self.MAX_PROCESSES)

		result = pool.map_async(self.evaluate_solution, population)
		
		pool.close()
		pool.join()

		for sol, res in zip(population, result.get()):
			fit, model = res
			sol.fitness = fit
			sol.phenotype = model
			sol.evaluated = True


	def replace(self, population, offspring):
		
		population += offspring
		population.sort(key=lambda x: x.fitness, reverse=self.MAXIMIZE)

		for _ in range(len(offspring)):
			population.pop()


	def execute(self, checkpoint=False):

		#if checkpoint:
		#	print('starting from checkpoint')
		#	population, evals = load_state()
		
		#if not population and not evals:
		#	print('starting from zero')
		self.population = self.create_population(self.POP_SIZE)
		self.evaluate_population(self.population)
		self.population.sort(key=lambda x: x.fitness, reverse=self.MAXIMIZE)

		self.evals = len(self.population)

		if self.DEBUG:
			for i, p in enumerate(self.population):
				print(i, p.fitness, p)

		print('<{}> evals: {}/{} \tbest so far: {}\tfitness: {}'.format(
			time.strftime('%x %X'), 
			self.evals, self.MAX_EVALS, 
			self.population[0].genotype, 
			self.population[0].fitness)
		)

		self.save_state()

		while self.evals < self.MAX_EVALS:
			parents = self.selection.execute(self.population)
			offspring_pop = []

			for _ in self.population:
				offspring = self.crossover.execute(parents)
				self.mutation.execute(offspring)
				self.prune.execute(offspring)
				self.duplication.execute(offspring)
				offspring_pop += offspring

			self.evaluate_population(offspring_pop)
			self.replace(self.population, offspring_pop)

			self.evals += len(offspring_pop)

			if self.DEBUG:
				for i, p in enumerate(self.population):
					print(i, p.fitness, p)

			print('<{}> evals: {}/{} \tbest so far: {}\tfitness: {}'.format(
				time.strftime('%x %X'), 
				self.evals, self.MAX_EVALS, 
				self.population[0].genotype, 
				self.population[0].fitness)
			)
			
			#save_state(evals, population)

		return self.population[0]


	def save_state(self):

		args = self.__dict__
		for key in args.keys():
			if key in ['selection', 'crossover', 'mutation', 'prune', 'duplication']:
				print(key, args[key].__dict__)

			elif key not in ['population', 'problem']:
				print(key, args[key])

		#folder = checkpoint.ckpt_folder
		#if not os.path.exists(folder): os.mkdir(folder)
		#checkpoint.save_args(args, os.path.join(folder, 'args_{}.ckpt'.format(evals)))
		#checkpoint.save_population(population, os.path.join(folder, 'pop_{}.ckpt'.format(evals)))



def load_state(args_file=None, pop_file=None):
	''' loads the state stored in both args file and pop file
		if one is None, the default behavior is to try to load the 
		most recent one
	'''
	global MAX_EVALS, CROSS_RATE, MUT_RATE, PRUN_RATE, DUPL_RATE, MINIMIZE

	folder = checkpoint.ckpt_folder

	pop_files = glob.glob(os.path.join(folder, 'pop_*'))
	for i, file in enumerate(pop_files):
		m = re.match('\\S+_([\\d]+).ckpt', file)
		id = int(m.group(1)) if m else 0
		pop_files[i] = {'id': id, 'file': file}

	arg_files = glob.glob(os.path.join(folder, 'args_*'))
	for i, file in enumerate(arg_files):
		m = re.match('\\S+_([\\d]+).ckpt', file)
		id = int(m.group(1)) if m else 0
		arg_files[i] = {'id': id, 'file': file}

	if pop_files == [] or arg_files == []:
		return None, None

	pop_files.sort(key=lambda x: x['id'], reverse=True)
	pop_file = pop_files[0]['file']
	population = checkpoint.load_population(pop_file)

	arg_files.sort(key=lambda x: x['id'], reverse=True)
	args_file = arg_files[0]['file']
	args = checkpoint.load_args(args_file)

	#POP_SIZE = args['POP_SIZE'] 
	#args['MIN_GENES']
	#args['MAX_GENES']
	#args['MAX_PROCESSES']
	
	print('CROSS_RATE set to', CROSS_RATE)
	CROSS_RATE = args['CROSS_RATE']
	
	print('MUT_RATE set to', MUT_RATE)
	MUT_RATE = args['MUT_RATE']

	print('PRUN_RATE set to', PRUN_RATE)
	PRUN_RATE = args['PRUN_RATE'] 

	print('DUPL_RATE set to', DUPL_RATE)
	DUPL_RATE = args['DUPL_RATE']

	print('MINIZE set to', MINIMIZE)
	MINIMIZE = args['MINIMIZE']

	evals = args['evals']
	print('evals set to', evals)

	MAX_EVALS = int(args['MAX_EVALS']) #temp
	print('MAX_EVALS set to', MAX_EVALS)

	return population, evals
