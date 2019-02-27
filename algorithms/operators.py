import numpy as np

# Selection

class TournamentSelection:

	''' Tournament Selection picks N random solutions,
		the best solution among these N is added to list of parents.
		The process is repeated for the number of desired parents.

		n_parents: the number os parents (default 2)
		t_size: number of solutions selected for the tournament (default 2)
		maximize: if the problem is a maximization problem (default False)
	'''

	def __init__(self, n_parents=2, t_size=2, maximize=False):
		self.n_parents = 2
		self.t_size = t_size
		self.maximize = maximize

	def __str__(self):
		return 'Tournament Selection'

	def execute(self, population):
		parents = []
		while len(parents) < self.n_parents:
			pool = []
			while len(pool) < self.t_size:
				temp = np.random.choice(population)
				if temp not in pool:
					pool.append(temp)
			pool.sort(key=lambda s: s.fitness, reverse=self.maximize)
			if pool[0] not in parents:
				parents.append(pool[0])
		return parents


# Crossover

class OnePointCrossover:

	''' One Point Crossover combines two solutions into one new by
		combining the first half of the first parent solution and
		the second part of the second parent solution

		cross_rate: chance to apply the operator
	'''

	def __init__(self, cross_rate):
		self.cross_rate = cross_rate

	def __str__(self):
		return 'One Point Crossover'

	def execute(self, parents):
		off1 = parents[0].copy()
		off2 = parents[1].copy()

		if np.random.rand() < self.cross_rate:
			p1 = off1.genotype[:]
			p2 = off2.genotype[:]
			min_len = min(len(p1), len(p2))
			cut = np.random.randint(0, min_len)
			off1.genotype = np.concatenate((p1[:cut], p2[cut:]))
		return [off1]


# Mutation

class PointMutation:

	''' Point Mutation changes a list of solutions by selecting a random
		point and generating a new value for that position (repeate for
		each solution)

		mut_rate: chance to apply the operator
		min_value: min possible value for the solution
		max_value: max possible value for the solution
	'''

	def __init__(self, mut_rate, min_value=0, max_value=1):
		self.mut_rate = mut_rate
		self.min_value = min_value
		self.max_value = max_value

	def __str__(self):
		return 'Point Mutation'

	def execute(self, offspring):
		if np.random.rand() < self.mut_rate:
			for off in offspring:
				index = np.random.randint(0, len(off.genotype))
				off.genotype[index] = np.random.randint(
					self.min_value, self.max_value)


# Prune

class GEPrune:

	'''
	'''

	def __init__(self, prun_rate):
		self.prun_rate = prun_rate

	def __str__(self):
		return 'Prune'

	def execute(self, offspring):
		if np.random.rand() < self.prun_rate:
			for off in offspring:
				if len(off.genotype) <= 1:
					return
					#if self.DEBUG: print('[prune] one gene, not applying:', off.genotype)
					#continue
				cut = np.random.randint(1, len(off.genotype))
				off.genotype = off.genotype[:cut]


# Duplication

class GEDuplication:

	'''
	'''

	def __init__(self, dupl_rate):
		self.dupl_rate = dupl_rate

	def __str__(self):
		return 'Duplication'

	def execute(self, offspring):
		if np.random.rand() < self.dupl_rate:
			for off in offspring:
				if len(off.genotype) > 1:
					cut = np.random.randint(0, len(off.genotype))
				else:
					#if self.DEBUG: print('[duplication] one gene, setting cut to 1:', off)
					cut = 1
				genes = off.genotype
				off.genotype = np.concatenate((genes, genes[:cut]))
#