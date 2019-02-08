from solutions import Solution

class BaseEvolutionaryAlgorithm:

	MAX_PROCESSES = 1

	def create_solution(self):
		raise NotImplementedError('Not implemented yet.')

	def evaluate_solution(self):
		raise NotImplementedError('Not implemented yet.')

	def selection(self, num_selections):
		raise NotImplementedError('Not implemented yet.')

	def pertubation(self, solution):
		raise NotImplementedError('Not implemented yet.')

	def execute(self):
		raise NotImplementedError('Not implemented yet.')


class RandomSearch(BaseEvolutionaryAlgorithm):

	problem = None
	best = None

	MIN_VALUE = 0
	MAX_VALUE = 1
	MIN_SIZE = 1
	MAX_SIZE = 10

	MAX_EVALS = 100
	MAXIMIZE = True

	def create_solution(self, min_size, max_size=None, min_value=0, max_value=1):

		values = rand.randint(min_value, max_value, rand.randint(min_size, max_size))
		return Solution(values)


	def 


	def execute(self):

		evals = 0

		while evals < self.MAX_EVALS:

			population = []
			for _ in range(self.MAX_PROCESSES):
				solution = self.create_solution(
					self.MIN_SIZE, self.MAX_SIZE, 
					self.MIN_VALUE, self.MAX_VALUE)
				population.append(solution)

			pool = Pool(processes=self.MAX_PROCESSES)

			result = pool.map_async(self.evaluate_solution, population)

			pool.close()
			pool.join()

			for sol, res in zip(population, result.get()):
				fit, model = res
				sol.fitness = fit
				sol.phenotype = model
				sol.evaluated = True

			if not best: population.append(best)
			population.sort(key=lambda x: x.fitness, reverse=self.MAXIMIZE)

			best = population[0].copy(deep=True)
			print('best', best.fitness)
