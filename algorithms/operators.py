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

	def __init__(self, n_parents = 2, t_size=2, maximize=False):
		self.n_parents = 2
		self.t_size = t_size
		self.maximize = maximize

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


# Mutation

