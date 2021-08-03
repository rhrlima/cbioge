import logging, sys, platform, calendar, time, os, codecs
from cbioge.utils.coreutil import contains, create_dir
import numpy as np
from cbioge.grammars.grammar import Grammar
from cbioge.problems.gcnProblem import GCNProblem
from cbioge.problems.multipleProblem import MultipleProblem
from cbioge.algorithms.selection import SimilaritySelection, TournamentSelection
from cbioge.algorithms.crossover import OnePointCrossover, DSGECrossover, DSGEGeneCrossover
from cbioge.algorithms.mutation import PointMutation, DSGEMutation, DSGETerminalMutation, DSGENonTerminalMutation
from cbioge.algorithms.mutation import DSGENonterminalMutation
from cbioge.algorithms.operators import ElitistReplacement, HalfAndHalfOperator
from cbioge.algorithms.dsge import GrammaticalEvolution
from cbioge.utils import checkpoint as ckpt
from cbioge.utils.experiments import check_os

logging.basicConfig(level=logging.INFO)

mandatory_params = ['dataset_path', 'dataset', 'output', 'grammar']

class GEEvolutionRunner():
    def __init__(self, args):
        logging.info(f":: GEEvolutionRunner Params: {args}")
        try:
            if not contains(mandatory_params, args.keys()):
                raise RuntimeError(f"We need mandatory params: f{mandatory_params}")

            none_mandatory_params = []

            for mp in mandatory_params:
                if args[mp] == None:
                    none_mandatory_params.append(mp)

            if len(none_mandatory_params)>0:
                raise RuntimeError(f"We need mandatory params: f{none_mandatory_params}")

            self.dataset_path = args['dataset_path']
            self.dataset = args['dataset']
            self.output = args['output']
            self.grammar_path = args['grammar']
            if 'verbose' in args:
                self.verbose = args['verbose']
            else:
                logging.info(":: No verbose param found, using default (1 - all logs are showed) ")
                self.verbose = 1
            
            if 'problem' in args:
                self.problem_model = args['problem']
            else:
                self.problem_model = GCNProblem


            if 'training' in args:
                self.training = args['training']
            else:
                logging.info(":: No training param found, using 'True'")
                self.training = True

            if 'epochs' in args:
                self.epochs = args['epochs']
            else:
                logging.info(":: No epochs to Neural Network param found, using default (5)")
                self.epochs = 5

            if 'batch' in args:
                self.batch = args['batch']
            else:
                logging.info(":: No batch param found, using default (1)")
                self.batch = 1

            if 'switch_mutation' in args:
                self.switch_mutation = args['switch_mutation']
            else:
                logging.info(":: No switch_mutation param found, using default (False)")
                self.switch_mutation = False

            if 'workers' in args:
                self.workers = args['workers']
            else:
                logging.info(":: No workers param found, using default (1)")
                self.workers = 1

            if 'multiprocess' in args:
                self.multiprocess = args['multiprocess']
            else:
                logging.info(":: No multiprocess param found, using default (True)")
                self.multiprocess = True

            if 'metrics' in args:
                self.metrics = args['metrics']
            else:
                logging.info(":: No metrics param found, using default (acc)")
                self.metrics = ['acc']


            if 'pop' in args:
                self.pop = args['pop']
            else:
                logging.info(":: No pop (population size) param found, using default (10)")
                self.pop = 20

            if 'gen' in args:
                self.gen = args['gen']
            else:
                self.gen = 2

            self.evals = self.pop*self.gen

            logging.info(f":: Generations to GE: {self.gen}")
            logging.info(f":: Evaluations to GE: {self.evals} (pop*gen)")


            if 'crossrate' in args:
                self.crossrate = args['crossrate']
            else:
                logging.info(":: No crossrate (rate of crossover) param found, using default (0.6)")
                self.crossrate = 0.6

            self.elitismrate = 0.25
            if 'elitismrate' in args:
                self.elitismrate = args['elitismrate']

            if 'mutrate' in args:
                self.mutrate = args['mutrate']
            else:
                logging.info(":: No mutrate (rate of mutation) param found, using default (0.4)")
                self.mutrate = 0.4

            if 'checkpoint' in args:
                self.checkpoint = True
                self.checkpoint_folder = args['checkpoint']
            else:
                self.checkpoint_folder = f"{self.output}/checkpoint"
                logging.info(f":: No checkpoint folder param found, using {self.checkpoint_folder}")

            if 'seed' in args:
                self.seed = args['seed']
            else:
                logging.info(":: No seed param found, using default (42) ")
                self.seed = 42
                np.random.seed(self.seed)

            if 'timelimit' in args:
                self.timelimit = args['timelimit']
            else:
                logging.info(":: No timelimit param found, using default (3600) ")
                self.timelimit = 3600


            if 't_size' in args:
                self.t_size = args['t_size']
            else:
                logging.info(":: No t_size (Tournament Selection size) param found, using default (5) ")
                self.t_size = 5

            if 'tournament_maximize' in args:
                self.tournament_maximize = args['t_size']
            else:
                logging.info(":: No t_size (tournament maximize) param found, using default (True) ")
                self.tournament_maximize = True

            if 'n_parents' in args:
                self.n_parents = args['n_parents']
            else:
                logging.info(":: No n_parents (number or parents) param found, using default (2) ")
                self.n_parents = 2

            if 'selection_method' in args:
                self.selection_method = args['selection_method']
            else:
                logging.info(":: No selection method param found, using default (TournamentSelection) ")
                self.selection_method = TournamentSelection

            if 'crossover_method' in args:
                self.crossover_method = args['crossover_method']
            else:
                logging.info(":: No crossover method param found, using default (DSGECrossover) ")
                self.crossover_method = DSGECrossover

            if 'mutation_method' in args:
                self.mutation_method = args['mutation_method']
            else:
                logging.info(":: No mutation method param found, using default (DSGEMutation) ")
                self.mutation_method = DSGEMutation

            if 'replacement_method' in args:
                self.replacement_method = args['replacement_method']
            else:
                logging.info(":: No replacement method param found, using default (ElitistReplacement) ")
                self.replacement_method = ElitistReplacement


            if 'replacement_rate' in args:
                self.replacement_rate = args['replacement_rate']
            else:
                logging.info(":: No replacement rate param found, using default (.25) ")
                self.replacement_rate = .25

            if 'replacement_maximize' in args:
                self.replacement_maximize = args['replacement_maximize']
            else:
                logging.info(":: No replacement maximize param found, using default (True) ")
                self.replacement_maximize = True

            self.ge_blocks = None
            if 'ge_blocks' in args:
                self.ge_blocks = args['ge_blocks']
            



        except:
            logging.error(":: Unexpected error: ", sys.exc_info()[0])
            raise

        self.timestamp = calendar.timegm(time.gmtime())
        self.grammar = None
        self.problem = None
        self.ge = None

        self.build()

    def build(self):
        # disable keras warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        check_os()
        logging.info(f":: Loading grammar: {self.grammar_path}")
        self.parser = Grammar(self.grammar_path)

        logging.info(f":: name: {self.parser.name} ")
        logging.info(f":: blocks: {self.parser.blocks} ")
        logging.info(f":: rules: {self.parser.rules} ")
        logging.info(f":: nonterm: {self.parser.nonterm} ")
        logging.info(f":: metrics: {self.metrics} ")

        if self.problem_model == GCNProblem:
            self.build_gcnProblem()

        elif self.problem_model == MultipleProblem:
            self.build_multipleProblem()

        return "SUCCESS"

    def build_multipleProblem(self):
        logging.info(f":: Building problem (MultipleProblem)")
        self.problem = MultipleProblem(parser=self.parser, verbose=True, timelimit = self.timelimit,
            epochs = self.epochs, workers = self.workers,
            multiprocessing = self.multiprocess,
            metrics = self.metrics)
        logging.info(f":: Reading dataset")
        self.problem.read_dataset_from_pickle(dataset=self.dataset, data_path=self.dataset_path)
        self.algorithm = GrammaticalEvolution(
            self.problem, self.parser, 
            pop_size=self.pop,
            max_evals=self.evals, 
            selection=TournamentSelection(t_size=self.t_size, maximize=self.tournament_maximize), 
            crossover=HalfAndHalfOperator(
                op1=DSGECrossover(cross_rate=self.crossrate), 
                op2=DSGENonterminalMutation(mut_rate=self.mutrate, parser=self.parser, end_index=4), 
                rate=0.6), 
            replacement=ElitistReplacement(rate=self.elitismrate, maximize=True), 
            verbose=True)

    def build_gcnProblem(self):
        logging.info(f":: Building problem (GCNProblem)")
        self.problem = GCNProblem(parser=self.parser, verbose=True, timelimit = self.timelimit,
            epochs = self.epochs, workers = self.workers,
            multiprocessing = self.multiprocess,
            metrics = self.metrics
            )

        logging.info(f":: Reading dataset")
        self.problem.read_dataset_from_pickle(dataset=self.dataset, data_path=self.dataset_path)


        logging.info(f":: Building algorithm")

        self.algorithm = GrammaticalEvolution(
            self.problem, self.parser, 
            pop_size=self.pop,
            max_evals=self.evals, 
            selection=TournamentSelection(t_size=self.t_size, maximize=self.tournament_maximize), 
            crossover=HalfAndHalfOperator(
                op1=DSGECrossover(cross_rate=self.crossrate), 
                op2=DSGENonterminalMutation(mut_rate=self.mutrate, parser=self.parser, end_index=4), 
                rate=0.6), 
            replacement=ElitistReplacement(rate=self.elitismrate, maximize=True), 
            verbose=True)

    def execute(self):
        self.population = self.algorithm.execute()
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return "SUCCESS"

