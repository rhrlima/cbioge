class BaseEvolutionaryAlgorithm:

	def __init__(self, problem):
		self.MAX_PROCESSES = 1
		self.MAXIMIZE = True

		self.problem = problem

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
		