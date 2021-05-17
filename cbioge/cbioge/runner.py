import logging, sys, platform, calendar, time, os, codecs
from cbioge.utils.coreutil import contains, create_dir
import numpy as np
from cbioge.grammars.grammar import BNFGrammar
from cbioge.problems.gcnProblem import GCNProblem
from cbioge.algorithms.selection import SimilaritySelection, TournamentSelection
from cbioge.algorithms.crossover import OnePointCrossover, DSGECrossover, DSGEGeneCrossover
from cbioge.algorithms.mutation import PointMutation, DSGEMutation, DSGETerminalMutation, DSGENonTerminalMutation
from cbioge.algorithms.operators import ElitistReplacement
from cbioge.algorithms.dsge import GrammaticalEvolution
from cbioge.utils import checkpoint as ckpt

logging.basicConfig(level=logging.INFO)

mandatory_params = ['dataset_path', 'dataset', 'output', 'grammar']

class GEEvolutionRunner():
    def __init__(self, args):
        logging.info(f"Params: {args}")
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

    def build(self):
        self.check()
        self.make_folders()
        self._partial_build()
        self.build_grammarEvolution()
        self.set_verbose()

    def set_verbose(self):
        if self.ge:
            self.ge.verbose = (self.verbose>0) # verbose 1 or higher
        if self.problem:
            self.problem.verbose = (self.verbose>1) # verbose 2 or higher
        if self.grammar:
            self.grammar.verbose = self.verbose>1

    def check(self)->bool:
        try:
            if platform.system() == 'Windows':
                limit_gpu_memory()
            return True
        except:
            return False

    def make_folders(self):
        logging.info(":: Creating output folder")
        create_dir(self.output)
        self.config_file_name = os.path.join(self.output, f"experiment_configuration_{self.timestamp}.txt")
        logging.info(f":: Creating configuration experiment file on {self.config_file_name }")
        self.config_file = codecs.open(self.config_file_name, "w")

    def build_grammar(self):
        try:
            self.grammar = BNFGrammar(self.grammar_path)
            return self.grammar
        except:
            logging.error(":: Unexpected error: ", sys.exc_info()[0])

        return None

    def build_problem(self):
        self.problem = None

        #Check if grammar exists
        if not self.grammar:
            self.build_grammar()

        if self.problem_model == GCNProblem:
            self.problem = GCNProblem(parser=self.grammar, verbose=True)

        self.problem.timelimit = self.timelimit
        self.problem.epochs = self.epochs
        self.problem.workers = self.workers
        self.problem.multiprocessing = self.multiprocess
        self.problem.metrics = self.metrics
        self.problem.read_dataset_from_pickle(dataset=self.dataset, data_path=self.dataset_path)
        
        return self.problem

    def build_selection(self):
        if self.selection_method == TournamentSelection or self.selection_method == 'TournamentSelection':
            self.selection = TournamentSelection(n_parents=self.n_parents, t_size=self.t_size, maximize=self.tournament_maximize)
        elif self.similaritySelection == SimilaritySelection or self.similaritySelection == 'SimilaritySelection':
            self.selection = SimilaritySelection(n_parents=self.n_parents, t_size=self.t_size, maximize=self.tournament_maximize)

        return self.selection
 
    def build_crossover(self):
        self.crossover = None
        if self.crossover_method == DSGECrossover or self.crossover_method == 'DSGECrossover':
            self.crossover = DSGECrossover(cross_rate=self.crossrate)
        else:
            self.crossover = self.crossover_method(cross_rate=self.crossrate)
        return self.crossover

    def build_mutation(self):
        self.mutation = None
        if self.mutation_method == DSGEMutation or self.mutation_method == 'DSGEMutation':
            self.mutation = DSGEMutation(mut_rate=self.mutrate, parser=self.grammar)
        else:
            self.mutation = self.mutation_method(mut_rate=self.mutrate, parser=self.grammar)

        return self.mutation


    def build_replacement(self):
        self.replacement = None
        if self.replacement_method == ElitistReplacement or self.replaclement_method == 'ElitistReplacement':
            self.replacement = ElitistReplacement(rate=self.replacement_rate, maximize=self.replacement_maximize)
        else:
            self.replacement = self.replaclement_method(rate=self.replacement_rate, maximize=self.replacement_maximize)

        return self.replacement

    def _partial_build(self):
        self.build_grammar()
        self.build_problem()
        self.build_selection()
        self.build_crossover()
        self.build_mutation()
        self.build_replacement()

    def build_grammarEvolution(self):
        self.ge = None
        if not self.problem:
            self._partial_build()

        self.ge = GrammaticalEvolution(self.problem, self.grammar)
        self.ge.selection = self.selection
        self.ge.crossover = self.crossover
        self.ge.mutation = self.mutation
        self.ge.replacement = self.replacement

        self.ge.pop_size = self.pop
        self.ge.max_evals = self.evals
    
        self.ge.switch_mutation = self.switch_mutation==1

        if self.ge_blocks:
            self.ge.blocks = self.ge_blocks

        ckpt.ckpt_folder = self.checkpoint_folder

        return self.ge

    def execute_ge(self):
        if not self.ge:
            logging.info(":: Build runner")
            self.build()

        self.population = self.ge.execute(self.checkpoint_folder)
        return self.population

