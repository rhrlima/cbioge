from cbioge.runner import GEEvolutionRunner
import logging

logging.basicConfig(level=logging.INFO)


class MultipleEvolution():
	def __init__(self, **args):
		logging.info(":: Creating MultipleEvolution")		
		self.runner = GEEvolutionRunner(args)


	def execute(self, **args)->str:
		
		logging.info(":: Creating GEEvolutionRunner")		
		
		self.runner = GEEvolutionRunner(args)

		self.verbose = 0

		if 'verbose' in args:
			self.verbose = args['verbose']

		if self.verbose>=1:
			logging.info(":: Vebose is on")

		logging.info(":: Executing GE in 3..2...1!")
		self.population = self.runner.execute()
		self.problem = self.runner.problem
		if self.verbose>=1:
			logging.info("Fitness\t\tPopulation")
			for s in self.population:
				logging.info(f"{s.fitness}\t\t{s}")
		logging.info(":: Finished")

		return 'SUCCESS'