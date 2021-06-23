from cbioge.runner import GEEvolutionRunner
import logging

logging.basicConfig(level=logging.INFO)


class GCNGEEvolution():

	def execute(self, **args)->str:
		
		logging.info(":: Creating GEEvolutionRunner")		
		
		self.runner = GEEvolutionRunner(args)


		self.verbose = 0

		if 'verbose' in args:
			self.verbose = args['verbose']

		if self.verbose>=1:
			logging.info(":: Vebose is on")

		logging.info(":: Build runner GEEvolutionRunner")
		self.runner.build()
		logging.info(":: Executing GE in 3..2...1!")
		self.population = self.runner.execute()
		self.problem = self.runner.problem
		if self.verbose>=1:
			logging.info("Fitness\t\tPopulation")
			for s in self.population:
				logging.info(f"{s.fitness}\t\t{s}")
		logging.info(":: Finished")

		return 'SUCCESS'

	def create_runner(self, dataset, dataset_path, output, grammar)->GEEvolutionRunner:
		runner = GEEvolutionRunner({'dataset':dataset, 'dataset_path':dataset_path, 'output':output, 'grammar':grammar})
		return runner
